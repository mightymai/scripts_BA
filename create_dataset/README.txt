1 frequency.R:
  	ADJUST: 'freq = 100' to set the threshold. This value will be the infimun, thus the smallest occurence allowed for the wordtokens
  	OUTPUT: 'dat_*freq*.csv' --> dataframe with only the desired wordtokens, their time intervalls (start and end) in corresponding file; sorted alphabetically 
  	HOW TO: Has to be exectued in folder with 'data.rds' (= labels for RedHen data)

2.1 extract_words.py:
	ADJUST: 'path' = path to original Data (broadcasts)
			'data_dir' = where to save created audio files
			'dat_path' = path to csv file from frequency.R (dat_*freq*.csv)
	OUTPUT: creates directories with audio files for training
	takes some time (number of iterations correspondig to number of rows of dat_*freq*.csv)

2.2 randomizer.py (OPTIONAL):
	For creating balanced datasets
	ADJUST: 'freq' = sum of number of audio files for each wordtoken (test + train)
			'path' = path of unbalanced data
			'balanced' = path to folder, where created data will be saved

3 randomizer_80_20.py:
	ADJUST: 'path' = path to created in step 2.1 or 2.2 (full balanced or unbalanced data)
			'dat_80_path' = path to folder, where train data will be saved (~80%)
			'dat_20_path' = path to folder, where test data will be saved (~20%)
	
			




