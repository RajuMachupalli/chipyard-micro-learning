// File: generators/mnist-accelerator/src/main/scala/MNISTAccelerator.scala
package mnist

import chisel3._
import chisel3.util._
import freechips.rocketchip.config._
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._

// Custom RISC-V instructions matching the C code operations
object MNISTInstructions {
  // fmac rd, rs1, rs2, rs3  -> rd = (rs1 * rs2) + rs3
  def FMAC = BitPat("b0000000??????????000?????0001011")
  
  // vfmac.vv vd, vs1, vs2, vs3 -> vd[i] = (vs1[i] * vs2[i]) + vs3[i]
  def VFMAC_VV = BitPat("b0000001??????????000?????0001011")
  
  // vfmac.mv vd, vs1, vs2, rs3 -> vd[i] = (vs1[i] * vs2[i]) + vd[i]
  def VFMAC_MV = BitPat("b0000010??????????000?????0001011")
}

// Parameters matching the C implementation needs
case class MNISTAcceleratorParams(
  vectorLength: Int = 8,     // Process 8 MACs in parallel
  dataWidth: Int = 32,       // 32-bit floats as in C code
  maxLayerSize: Int = 128,   // Largest layer (HIDDEN1_SIZE)
  numLayers: Int = 3,        // Three layers in the network
  pipelineStages: Int = 3    // Pipeline depth for MAC
)

// Floating-point MAC unit matching matrix_multiply_add function
class FloatingPointMAC(dataWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(dataWidth.W))      // input
    val b = Input(UInt(dataWidth.W))      // weight
    val c = Input(UInt(dataWidth.W))      // bias/accumulator
    val result = Output(UInt(dataWidth.W))
    val valid = Input(Bool())
    val ready = Output(Bool())
  })

  // Three-stage pipeline matching typical FPU timing
  val s1_valid = RegNext(io.valid, false.B)
  val s2_valid = RegNext(s1_valid, false.B)
  val s3_valid = RegNext(s2_valid, false.B)
  
  // Stage 1: Multiply
  val product = RegEnable(io.a * io.b, io.valid)
  
  // Stage 2: Add
  val sum = RegEnable(product + io.c, s1_valid)
  
  // Stage 3: Output
  val result = RegEnable(sum, s2_valid)
  
  io.result := result
  io.ready := s3_valid
}

// Vector MAC unit for processing multiple operations in parallel
class VectorMACUnit(implicit p: Parameters) extends Module {
  val params = p(MNISTAcceleratorKey)
  val io = IO(new Bundle {
    // Control signals
    val valid = Input(Bool())
    val accumulate = Input(Bool())  // If true, add to existing result
    
    // Vector inputs - matching layer computations
    val inputs = Input(Vec(params.vectorLength, UInt(params.dataWidth.W)))
    val weights = Input(Vec(params.vectorLength, UInt(params.dataWidth.W)))
    val bias = Input(UInt(params.dataWidth.W))
    
    // Vector output
    val results = Output(Vec(params.vectorLength, UInt(params.dataWidth.W)))
    val ready = Output(Bool())
  })

  // Instantiate parallel MAC units
  val macUnits = Seq.fill(params.vectorLength)(Module(new FloatingPointMAC(params.dataWidth)))
  
  // Connect inputs to MAC units
  for (i <- 0 until params.vectorLength) {
    macUnits(i).io.a := io.inputs(i)
    macUnits(i).io.b := io.weights(i)
    macUnits(i).io.c := Mux(io.accumulate, io.results(i), io.bias)
    macUnits(i).io.valid := io.valid
  }
  
  // Collect results
  for (i <- 0 until params.vectorLength) {
    io.results(i) := macUnits(i).io.result
  }
  
  io.ready := macUnits(0).io.ready  // All units have same timing
}

// Main accelerator targeting the three layers
class MNISTAccelerator(implicit p: Parameters) extends RoCC()(p) {
  val params = p(MNISTAcceleratorKey)
  
  // Instantiate vector MAC unit
  val vectorMAC = Module(new VectorMACUnit)
  
  // State machine for layer processing
  val s_idle :: s_process_layer :: s_accumulate :: Nil = Enum(3)
  val state = RegInit(s_idle)
  
  // Registers for layer processing
  val inputPtr = Reg(UInt(64.W))      // Pointer to input array
  val weightPtr = Reg(UInt(64.W))     // Pointer to weight matrix
  val outputPtr = Reg(UInt(64.W))     // Pointer to output array
  val biasValue = Reg(UInt(32.W))     // Current bias value
  
  val inputSize = Reg(UInt(16.W))     // Current layer input size
  val outputSize = Reg(UInt(16.W))    // Current layer output size
  val inputIdx = Reg(UInt(16.W))      // Current input index
  val outputIdx = Reg(UInt(16.W))     // Current output index
  
  // Accumulator for partial results
  val accumulator = Reg(Vec(params.maxLayerSize, UInt(params.dataWidth.W)))
  
  // Performance counter
  val cycleCount = Reg(UInt(32.W))
  val macCount = Reg(UInt(32.W))
  
  // Decode custom instructions
  val cmd = io.cmd.bits.inst
  val funct = cmd.funct
  val opcode = cmd.opcode
  
  val doFMAC = io.cmd.valid && (cmd === MNISTInstructions.FMAC)
  val doVectorMAC = io.cmd.valid && (cmd === MNISTInstructions.VFMAC_VV)
  val doMatrixMAC = io.cmd.valid && (cmd === MNISTInstructions.VFMAC_MV)
  
  // Default connections
  vectorMAC.io.valid := false.B
  vectorMAC.io.accumulate := false.B
  vectorMAC.io.inputs := DontCare
  vectorMAC.io.weights := DontCare
  vectorMAC.io.bias := biasValue
  
  io.cmd.ready := state === s_idle
  io.resp.valid := false.B
  io.resp.bits.data := 0.U
  io.busy := state =/= s_idle
  io.interrupt := false.B
  
  // Memory interface (simplified - would need proper TileLink in real implementation)
  io.mem.req.valid := false.B
  io.mem.req.bits := DontCare
  
  // Main state machine
  switch(state) {
    is(s_idle) {
      when(doVectorMAC) {
        // Start vector MAC operation
        // rs1 = input pointer, rs2 = weight pointer, rd = output pointer
        inputPtr := io.cmd.bits.rs1
        weightPtr := io.cmd.bits.rs2
        outputPtr := io.cmd.bits.inst.rd
        
        // Extract layer dimensions from instruction
        inputSize := io.cmd.bits.inst.imm(15, 0)
        outputSize := io.cmd.bits.inst.imm(31, 16)
        
        inputIdx := 0.U
        outputIdx := 0.U
        cycleCount := 0.U
        macCount := 0.U
        
        state := s_process_layer
      }.elsewhen(doFMAC) {
        // Single MAC operation for compatibility
        val result = (io.cmd.bits.rs1 * io.cmd.bits.rs2) + io.cmd.bits.inst.rd
        io.resp.valid := true.B
        io.resp.bits.data := result
      }
    }
    
    is(s_process_layer) {
      // Process layer computation in chunks of vectorLength
      vectorMAC.io.valid := true.B
      
      // Load inputs and weights (simplified - real implementation needs memory interface)
      for (i <- 0 until params.vectorLength) {
        vectorMAC.io.inputs(i) := inputPtr + (inputIdx + i.U) * 4.U  // Simplified
        vectorMAC.io.weights(i) := weightPtr + ((inputIdx * outputSize + outputIdx) + i.U) * 4.U
      }
      
      // Track progress
      when(vectorMAC.io.ready) {
        macCount := macCount + params.vectorLength.U
        outputIdx := outputIdx + params.vectorLength.U
        
        when(outputIdx >= outputSize) {
          outputIdx := 0.U
          inputIdx := inputIdx + 1.U
          
          when(inputIdx >= inputSize) {
            // Layer complete
            state := s_idle
            io.resp.valid := true.B
            io.resp.bits.data := macCount  // Return number of MACs performed
          }
        }
      }
      
      cycleCount := cycleCount + 1.U
    }
  }
  
  // Performance monitoring outputs (for profiling)
  when(state === s_idle && io.cmd.valid && io.cmd.bits.inst.funct === "b1111111".U) {
    // Special instruction to read performance counters
    io.resp.valid := true.B
    io.resp.bits.data := Cat(cycleCount, macCount)
  }
}

// RoCC interface wrapper
class MNISTRoCCAccelerator(implicit p: Parameters) extends LazyRoCC()(p) {
  override lazy val module = new MNISTRoCCAcceleratorModule(this)
}

class MNISTRoCCAcceleratorModule(outer: MNISTRoCCAccelerator) 
    extends LazyRoCCModuleImp(outer) {
  val accelerator = Module(new MNISTAccelerator()(p))
  accelerator.io <> io
}

// Configuration key
case object MNISTAcceleratorKey extends Field[MNISTAcceleratorParams](
  MNISTAcceleratorParams()
)

// Mixin trait for adding to Rocket core
class WithMNISTAccelerator extends Config((site, here, up) => {
  case MNISTAcceleratorKey => MNISTAcceleratorParams(
    vectorLength = 8,      // Process 8 MACs in parallel
    dataWidth = 32,        // 32-bit floats
    maxLayerSize = 128,    // HIDDEN1_SIZE
    numLayers = 3,         // Three layers
    pipelineStages = 3     // Typical FPU pipeline depth
  )
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      val mnist = LazyModule(new MNISTRoCCAccelerator()(p))
      mnist
    }
  )
})