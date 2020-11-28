rule data:
    input:
    output:
    shell:
        """ 
        mkdir -p data/external
        mkdir -p data/interim
        mkdir -p data/processed
        mkdir -p data/raw
        
        mkdir -p data/external/spell-ru-eval
        wget http://www.dialog-21.ru/media/3837/source_sents.txt -O data/external/spell-ru-eval/train_source.txt
        wget http://www.dialog-21.ru/media/3836/corrected_sents.txt -O data/external/spell-ru-eval/train_corrected.txt
        wget http://www.dialog-21.ru/media/3838/test_sample_testset.txt -O data/external/spell-ru-eval/test_source.txt
        wget http://www.dialog-21.ru/media/3835/corr_sample_testset.txt -O data/external/spell-ru-eval/test_corrected.txt
        """
