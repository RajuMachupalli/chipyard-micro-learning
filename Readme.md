RISC-V AI Chips: The 15-Minute Masterclass
A practical microcourse on building AI accelerators with Chipyard

Introduction: Remember Your First Slow MNIST?
Remember how slow your first MNIST neural network was? You loaded those 28×28 handwritten digit images, built a simple 3-layer network, hit "train"... and waited. Then you ran inference on a single digit and watched your CPU churn through 200,000 operations just to recognize one number.
If you profiled that code on a standard RISC-V processor, you'd see something brutal: 47 milliseconds per digit. That's only 21 digits per second. Your smartphone recognizes thousands of objects in real-time, but your carefully crafted neural network struggles with handwritten numbers.
Here's the thing: this isn't a software problem. It's a hardware mismatch.
Your challenge for the next 13 minutes: Let's make your MNIST inference 40x faster using custom RISC-V instructions with Chipyard. No PhD in computer architecture required—just the willingness to see how AI chips really work under the hood.
By the end, you'll understand why Google built the TPU, why Apple created the Neural Engine, and how you can build your own AI accelerator in an afternoon.

