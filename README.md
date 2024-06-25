"Test Parameters.pdf" reviews all parameters used to reproduce tests, and also supplemental material, such as specific test data.

Running tests from PopDescent ICLR 2024 Submission Paper:


FOR ALL TESTS:
- The file "Test Parameters.pdf" contains all notes on how to reproduce all tests in the paper
- all seeds are preloaded, you simply have to follow the commands listed below
- we ran all tests on an M1 silicon macbook, but we have requirements.txt that do not use the M1 chip as well
- all results are saved to a .csv file on the local desktop, but the direct filepath can be changed at the bottom of any test below
- In the .csv file, the first two fields listed are always the best returned test loss, and the best returned training loss



1) BENCHMARKS: Test each optimization method individually

PopDescent Benchmark Steps:
- In the anonymous_populationDescent folder, run the following instructions:

"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"

(to run FMNIST Benchmark without regularization): "python3 -m PD_FMNIST_Benchmark_without_regularization"
(to run FMNIST Benchmark with regularization): "python3 -m PD_FMNIST_Benchmark_with_regularization"
(to run FMNIST Benchmark without regularization): "python3 -m PD_CIFAR_Benchmark_without_regularization"
(to run FMNIST Benchmark with regularization): "python3 -m PD_CIFAR_Benchmark_with_regularization"



Basic Grid Search Benchmark Steps:
- In the anonymous_populationDescent folder, run the following instructions:

"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"

(to run FMNIST Benchmark without regularization): "python3 -m BasicGridSearch_FMNIST_Benchmark_without_regularization"
(to run FMNIST Benchmark with regularization): "python3 -m BasicGridSearch_FMNIST_Benchmark_with_regularization"
(to run CIFAR Benchmark without regularization): "python3 -m BasicGridSearch_CIFAR_Benchmark_without_regularization"
(to run CIFAR Benchmark with regularization): "python3 -m BasicGridSearch_CIFAR_Benchmark_with_regularization"



KT RandomSearch Benchmark Steps:
- In the anonymous_populationDescent folder, run the following instructions:

"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"

(to run FMNIST Benchmark without regularization): "python3 -m KT_RandomSearch_FMNIST_Benchmark_without_regularization"
(to run FMNIST Benchmark with regularization): "python3 -m KT_RandomSearch_FMNIST_Benchmark_with_regularization"
(to run CIFAR Benchmark without regularization): "python3 -m KT_RandomSearch_CIFAR_Benchmark_without_regularization"
(to run CIFAR Benchmark with regularization): "python3 -m KT_RandomSearch_CIFAR_Benchmark_with_regularization"



ESGD Benchmark Steps:
- In the anonymous_populationDescent folder, run the following instructions:

(to run FMNIST Benchmark without regularization):
- "cd ESGD_FMNIST_Benchmark"
- "python3 -m venv m1"
- "source m1/bin/activate"
- "pip3 install -r requirements.txt"
- "python3 -m esgd-ws -a "esgd""

(to run CIFAR Benchmark without regularization):
- "cd ESGD_CIFAR_Benchmark"
- "python3 -m venv m1"
- "source m1/bin/activate"
- "pip3 install -r requirements.txt"
- "python3 -m esgd-ws -a "esgd""




2) ABLATION TESTS: Test each study individually
- In the anonymous_populationDescent folder, run the following instructions:

"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"

(to run FMNSIT Ablation with regularization, without randomization): "python3 -m ablation_withReg_noRand_FMNIST"
(to run FMNSIT Ablation with regularization, with randomization): "python3 -m ablation_withReg_withRand_FMNIST"
(to run FMNSIT Ablation without regularization, without CV): "python3 -m ablation_noReg_noCV_FMNIST"
(to run FMNSIT Ablation without regularization, with CV): "python3 -m ablation_noReg_withCV_FMNIST"




3) PARAMETER SENSITIVITY TESTS: Test PopDescent and ESGD individually for iterations/learning rates
- In the anonymous_populationDescent folder, run the following instructions:

(to run PD on CIFAR while changing the iterations):

"cd PD_CIFAR_Sensitivity"
"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"
"python3 -m PD_CIFAR_Sensitivity_it"

(to run PD on CIFAR while changing the learning rate):

"cd PD_CIFAR_Sensitivity"
"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements_m1_metal.txt"
"python3 -m PD_CIFAR_Sensitivity_lr"


(to run ESGD on CIFAR while changing the iterations):

"cd ESGD_CIFAR_Sensitivity"
"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements.txt"
"python3 -m esgd-ws-it -a "esgd""

(to run ESGD on CIFAR while changing the learning rate):

"cd ESGD_CIFAR_Sensitivity"
"python3 -m venv m1"
"source m1/bin/activate"
"pip3 install -r requirements.txt"
"python3 -m esgd-ws-lr -a "esgd""























