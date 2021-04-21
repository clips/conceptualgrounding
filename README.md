# Conceptual grounding constraints for truly robust biomedical name representations

This directory contains source code for the following paper:

`Conceptual Grounding Constraints for Truly Robust Biomedical Name Representations.` \
Pieter Fivez, Simon Šuster and Walter Daelemans. *EACL*, 2021.

If you use this code, please cite:

    @inproceedings{fivez-etal-2021-conceptual,
    title = "Conceptual Grounding Constraints for Truly Robust Biomedical Name Representations", <br/>
    author = "Fivez, Pieter  and
      Suster, Simon  and
      Daelemans, Walter",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "2440--2450"}


The fastText model used in the paper can be downloaded from the following link: 
https://drive.google.com/file/d/1B07lc3eeW_zughHguugLBR4iJYQj_Wxz/view?usp=sharing

## License

GPL-3.0

## Requirements

All requirements are listed in **requirements.txt**. 

You can run `pip install -r requirements.txt`, preferably in a virtual environment.

## Data

For convenience, we only provide our adaptation of the openly available MedMentions corpus. \
The source files for this corpus can be found at https://github.com/chanzuckerberg/MedMentions.

The script **data/extract_medmentions.py** has used these source files to create **data/medmentions.json**.

## Data

We provide 2 scripts to run experiments from the paper. 

**main_dan.py** trains and evaluates the DAN encoder on **data/medmentions.json**. \
Please run `python main_dan.py --help` to see the options, or check the script. \
The default parameters are the best parameters reported in our paper.

**main_bne.py** trains and evaluates our counterpart implementation of the BNE encoder, as described in:

    @inproceedings{phan-etal-2019-robust,
    title = "Robust Representation Learning of Biomedical Names",
    author = "Phan, Minh C.  and
      Sun, Aixin  and
      Tay, Yi",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    pages = "3275--3285"}
    
Please run `python main_bne.py --help` to see the options, or check the script. \
The default parameters are the best parameters reported in our paper.

