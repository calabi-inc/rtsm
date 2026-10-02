# Use Cases

What `rtsm eval` says about public recordings nobody at Calabi made. Each page gives the command, what the bag reader decided (which topic plays which role, where the poses come from, whether the depth is registered), the report's numbers with their same-input floors over three runs, and an honest reading. The numbers are produced by the released code on the released defaults; the pages are regenerated for each release.

| recording | what it is | what it shows |
|---|---|---|
| [TUM fr3 / long_office_household](tum-fr3-long-office.md) | a handheld Kinect loop through an office with a desk revisited (2012, 640×480, mocap poses) | the full single-bag report on a loop with revisits |
| [TUM fr1 / desk](tum-fr1-desk.md) | a short handheld sweep over a desk (2012, 640×480, mocap poses) | a near-stationary scene: dense observations of few objects |
| [NVIDIA r2b_cafe](nvidia-r2b-cafe.md) | a RealSense D455 sequence from an autonomous-robot benchmark (2023) | what a bag needs and why RTSM refuses rather than guesses |

Two caveats apply everywhere. The TUM sequences are 2012 Kinect data at 640×480 with the camera's own noise, so the absolute numbers describe that sensor as much as the memory. Every detection and label comes from RTSM's packaged default detector (`grounded_sam2`, the public indoor vocabulary); the report attributes its numbers to it, and a different detector gives different numbers (see [Your own detector](../guides/eval.md#your-own-detector)).

Datasets: TUM RGB-D (CC BY 4.0; Sturm, Engelhard, Endres, Burgard, Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012), NVIDIA r2b Dataset 2023 (CC BY 4.0). Nothing from either dataset is redistributed here; the pages hold the commands and the numbers.
