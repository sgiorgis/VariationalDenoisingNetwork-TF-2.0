# VDNet - TF 2.0

Reproduction of "Variational Denoising Network: Toward Blind Noise
Modeling and Removal" (NeurIPS 2019) in Tensorflow 2.0.

[Github] | [Paper]

### Prerequisites

Our projects needs all requirements found in requirements.txt file. 
You can download all the requirements using the command
```
pip3 install -r requirements.txt
```

### Real-world image denoising task
You need to download the [SIDD Medium Dataset] we used to train our models for this task.
The SIDD Medium dataset is similar to SIDD Small dataset except that SIDD Medium consists of 320 image pairs (noisy and ground-truth),
two image pairs from each scene instance.
For validation you need to download the [Noisy sRGB data] and the [Ground-truth raw-RGB data] in ./benchmark_datasets

#### Validation data preprocessing
You need to preprocess the validation data as our model needs hdf5 files instead of mat files.
```
python3 datasets/validation_data_processing/validation_data_processing.py --data_directory ./benchmark_datasets --clean_mat_file ValidationGtBlocksSrgb.mat --noisy_mat_file ValidationNoisyBlocksSrgb.mat
```
This will generate the files ./datasets/ValidationGtBlocksSrgb.hdf5 and ./benchmark_datasets/ValidationNoisyBlocksSrgb.hdf5 
that will be used for validation. 

#### Train
```
python3 train/train.py --train_dataset_base_path ./benchmark_datasets/SIDD_Medium_Srgb --validation_clean_dataset_path ./benchmark_datasets/ValidationGtBlocksSrgb.hdf5 --validation_noisy_dataset_path  ./benchmark_datasets/ValidationNoisyBlocksSrgb.hdf5 --batches 5000 --batch_size 64 --checkpoint_directory ./benchmark_checkpoints
```
#### Validation
You can validate pre trained models or validate your trained model stored in checkpoints directory with
```
python3 test/test.py --validation_clean_dataset_path ./benchmark_datasets/ValidationGtBlocksSrgb.hdf5 --validation_noisy_dataset_path  ./benchmark_datasets/ValidationNoisyBlocksSrgb.hdf5 --batches 5000 --batch_size 64 --checkpoint_directory ./benchmark_checkpoints
```
#### Benchmark
To validate pre trained models or validate your trained model stored in checkpoints directory
to the official [Benchmark Noisy sRGB data] you need to download the data from the link provided
and move it to the ./benchmark_datasets folder
Then you can generate the denoised results that you need to upload to the website with
```
python3 test/test_benchmark.py  --noisy_mat_file ./benchmark_datasets/BenchmarkNoisyBlocksSrgb.mat --checkpoint_directory ./benchmark_checkpoints
```
This will generate the ./benchmark_datasets/SubmitSrgb.mat file contaiing the denoised images following the 
same structure as the ./benchmark_datasets/BenchmarkNoisyBlocksSrgb.mat file.

### Non-IID Gaussian image denoising task
For this task the validation and train processes are not connected since the validation 
datasets have images with different dimensions, thus we could not apply the same dataset generators.
Therefore you need to first train the whole network and then validate on the validation datasets.
#### Train
You need follow the instruction to download [Waterloo Exploration Database], CBSD432 and the CImageNet400 in [Github].
You need to download the datasets in ./simulation_datasets. 
```
python3 train/train_simulation.py --train_dataset_base_path ./simulation_datasets --batches 5000 --batch_size 64 --checkpoint_directory ./simulation_checkpoints
```
#### Validation
For validation you need to download the LIVE1, Set5 and the CBSD68 datasets. You can follow the instruction here [Github].
These are clean images so in order to add synthetic Non-IID Gaussian noise with different kernels, you need to run
```
python3 datasets/generate_noise.py --mode=non-iid --data_directory=./simulation_datasets --data_name=LIVE1 --extension=bmp
```
```
python3 datasets/generate_noise.py --mode=non-iid --data_directory=./simulation_datasets --data_name=Set5 --extension=bmp
```
```
python3 datasets/generate_noise.py --mode=non-iid --data_directory=./simulation_datasets --data_name=CBSD68 --extension=png
```

This will generate the files CBSD68_clean.hdf5, CBSD68_case1.hdf5, CBSD68_case2.hdf5, CBSD68_case3.hdf5 for CBSD68 dataset.
Same way it will generate the corresponding files for LIVE1 and Set5 as well.
Cases correspond to different kernels as described in the [Paper].

You can validate the three different kernels and print the validation_mse, validation_psnr, validation_ssim for CBSD68 dataset with 
```
python3 test/test_simulation.py --validation_clean_dataset_path ./simulation_datasets/CBSD68_clean.hdf5 --validation_noisy_dataset_path ./simulation_datasets/CBSD68_case1.hdf5 --checkpoint_directory ./simulation_checkpoints
```
```
python3 test/test_simulation.py --validation_clean_dataset_path ./simulation_datasets/CBSD68_clean.hdf5 --validation_noisy_dataset_path ./simulation_datasets/CBSD68_case2.hdf5 --checkpoint_directory ./simulation_checkpoints
```
```
python3 test/test_simulation.py --validation_clean_dataset_path ./simulation_datasets/CBSD68_clean.hdf5 --validation_noisy_dataset_path ./simulation_datasets/CBSD68_case3.hdf5 --checkpoint_directory ./simulation_checkpoints
```
Furthermore you can add --verbose=True to save the triplets of clean_image, noisy_image, denoised_image with matplotlib.
Following the same instructions you can validate the three different kernels for LIVE1 and Set5 datasets.

## Experiments
To showcase our work, we present our reproduction results on the Synthetic Non-I.I.D. Gaussian Noise.

|Cases            | Dataset   | Original | Reproduction |
|:----------------|:----------|:--------------| -----:|
| Case1           | SET5      | 30.39         | 30.38 |
|                 | LIVE1     | 29.22         | 29.28 |
|                 | BSD68     | 29.02         | 29.03 |
| Case2           | SET5      | 29.80         | 29.81 |
|                 | LIVE1     | 28.82         | 28.88 |
|                 | BSD68     | 28.67         | 28.69 |
| Case3           | SET5      | 29.74         | 29.81 |
|                 | LIVE1     | 28.65         | 28.70 |
|                 | BSD68     | 28.46         | 28.49 |

Once you have generated the synthetic noisy data in ./simulation_datasets, you can yield the same results using our pretrained model 
found in ./pretrained_synthetic_model by running:
```
python3 test/test_simulation.py --validation_clean_dataset_path ./simulation_datasets/CBSD68_clean.hdf5 --validation_noisy_dataset_path ./simulation_datasets/CBSD68_case3.hdf5 --checkpoint_directory ./pretrained_synthetic_model
```
## Acknowledgments

This work was initiated as a project of our master's level course titled 'Deep Learning, Advanced Course' @ KTH Stockholm, Sweden. We would like to thank the course staff for providing us with the necessary Google Cloud tickets to run our experiments.

[Paper]: https://arxiv.org/pdf/1908.11314v2.pdf
[Github]: https://github.com/zsyOAOA/VDNet
[SIDD Medium Dataset]: ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip
[Noisy sRGB data]: ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat
[Ground-truth raw-RGB data]: ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksRaw.mat
[Reproducibility Challenge @ NeurIPS 2019]: https://reproducibility-challenge.github.io/neurips2019/
[Waterloo Exploration Database]: https://ece.uwaterloo.ca/~k29ma/exploration/
[Benchmark Noisy sRGB data]: ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat