rule data:
    input:
    output:
    shell:
        """ 
        mkdir -p data/external
        mkdir -p data/interim
        mkdir -p data/processed
        mkdir -p data/raw
        """
