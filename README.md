# Uncanny Valley EEE

This is the source code for the experiment and data analysis performed in the [Investigating the Uncanny Valley Phenomenon Through the Temporal Dynamics of Neural Responses to Virtual Characters](https://arxiv.org/abs/2306.16233) published in the proceedings of the [2023 IEEE Conference on Games](https://2023.ieee-cog.org/).

The data collected and analysed in the article is available on Zenodo at [https://www.zenodo.org/record/7948158](https://www.zenodo.org/record/7948158). 
To execute the code in the repository it is first necessary to download the [Raw.zip](https://www.zenodo.org/record/7948158/files/Raw.zip) and [Scratch.zip](https://www.zenodo.org/record/7948158/files/Scratch.zip) files and decompress the content in the code root folder as the scripts expect the data to be contained in the Raw and Scratch folders.
If your OS is able to execute cURL and tar, you can run the *download_data.sh* script from the project's root folder.

Be aware the experiment runs in psychopy, which requires a C++ compiler that needs to be installed before running.
