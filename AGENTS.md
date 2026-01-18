Agents shall not ape humans (say "I", "please", "thank you", discuss emtions or whatever - agents shall stay impersonal and factual). Where "I did this" could be used ("I" referring to the agent) use the passive voice ("This was done")

Agents shall look in doc and refer to the CODEX_MANIFEST.md for further instructions and to the ActionScript_codebase.txt to understand how the original game worked (in detail) in order to remain faithful to it.

The LinuxBox on which I am training is  a Lenovo ThinkStation P3 Ultra  (CPU : Intel Core i7-14700, GPU : NVIDIA AD107GL [RTX 2000 / 2000E Ada Generation] (rev a1), OS : Ubuntu 6.8.0-48-generic, accessed via SSH from my Windows 11 machine - by default never via GUI, only SSH).
I use
- GNU bash, version 5.1.16(1)-release
- python 3.11.14
- conda 25.3.0.
- git version 2.34.1 with a PAT (style : https://${GH_TOKEN}@...)
- NVIDIA-SMI 575.51.03; CUDA Version: 12.9; Cuda compilation tools release 12.4, V12.4.131

lscpu -C yields :

NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       48K     768K   12 Data            1    64        1             64
L1i       32K       1M    8 Instruction     1    64        1             64
L2         2M      28M   16 Unified         2  2048        1             64
L3        33M      33M   11 Unified         3 49152        1             64

lscpu -e=CPU,CORE,MAXMHZ,MINMHZ,CACHE | sort -n
  0    0 5300,0000 800,0000 0:0:0:0
CPU CORE    MAXMHZ   MINMHZ L1d:L1i:L2:L3
  1    0 5300,0000 800,0000 0:0:0:0
  2    1 5300,0000 800,0000 4:4:1:0
  3    1 5300,0000 800,0000 4:4:1:0
  4    2 5300,0000 800,0000 8:8:2:0
  5    2 5300,0000 800,0000 8:8:2:0
  6    3 5300,0000 800,0000 12:12:3:0
  7    3 5300,0000 800,0000 12:12:3:0
  8    4 5400,0000 800,0000 16:16:4:0
  9    4 5400,0000 800,0000 16:16:4:0
 10    5 5400,0000 800,0000 20:20:5:0
 11    5 5400,0000 800,0000 20:20:5:0
 12    6 5300,0000 800,0000 24:24:6:0
 13    6 5300,0000 800,0000 24:24:6:0
 14    7 5300,0000 800,0000 28:28:7:0
 15    7 5300,0000 800,0000 28:28:7:0
 16    8 4200,0000 800,0000 32:32:8:0
 17    9 4200,0000 800,0000 33:33:8:0
 18   10 4200,0000 800,0000 34:34:8:0
 19   11 4200,0000 800,0000 35:35:8:0
 20   12 4200,0000 800,0000 36:36:9:0
 21   13 4200,0000 800,0000 37:37:9:0
 22   14 4200,0000 800,0000 38:38:9:0
 23   15 4200,0000 800,0000 39:39:9:0
 24   16 4200,0000 800,0000 40:40:10:0
 25   17 4200,0000 800,0000 41:41:10:0
 26   18 4200,0000 800,0000 42:42:10:0
 27   19 4200,0000 800,0000 43:43:10: 