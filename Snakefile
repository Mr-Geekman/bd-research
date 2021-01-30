rule data:
    input:
    output:
    shell:
        """ 
        # create directories for data
        mkdir -p data/external
        mkdir -p data/interim
        mkdir -p data/processed
        mkdir -p data/raw
        
        # load training data
        mkdir -p data/external/spell_ru_eval
        wget http://www.dialog-21.ru/media/3837/source_sents.txt -O data/external/spell_ru_eval/train_source.txt
        wget http://www.dialog-21.ru/media/3836/corrected_sents.txt -O data/external/spell_ru_eval/train_corrected.txt
        wget http://www.dialog-21.ru/media/3838/test_sample_testset.txt -O data/external/spell_ru_eval/test_source.txt
        wget http://www.dialog-21.ru/media/3835/corr_sample_testset.txt -O data/external/spell_ru_eval/test_corrected.txt

        # load vocabulary
        wget http://files.deeppavlov.ai/deeppavlov_data/vocabs/russian_words_vocab.dict.gz -P data/external/
        gzip -d data/external/russian_words_vocab.dict.gz

        # load rubert model
        wget http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz -P models/
        tar -xf models/rubert_cased_L-12_H-768_A-12_pt.tar.gz -C models/
        mv models/rubert_cased_L-12_H-768_A-12_pt models/rubert

        # load coversational rubert model
        wget http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12_pt.tar.gz -P models/
        tar -xf models/ru_conversational_cased_L-12_H-768_A-12_pt.tar.gz -C models/
        mv models/ru_conversational_cased_L-12_H-768_A-12_pt models/conversational_rubert
        """
