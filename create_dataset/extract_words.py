import os
import pandas
import scipy.io.wavfile

############## ADJUST HERE ##############
path = '/home/mai/OG_data/well_aligned_clean_16000Hz'
data_dir = '/home/mai/BA_wuerdig/data_20/complete_audio_20/unbalanced'
csv_path = '/home/mai/BA_wuerdig/data_20/dat_20.csv'
#########################################

os.makedirs(data_dir, mode=0o755, exist_ok=True)

# Wort aus Tabelle --> Timestamps rausschreiben + Filename
table = pandas.read_csv(csv_path)

ii = 0
for _, row in table.iterrows():
	name, file_name, start, end = row[["wordtoken", "File", "start", "end"]]

	rate, data = scipy.io.wavfile.read(os.path.join(path, file_name))

	# begin timestamp
	# hier wird alles vor 'start' abgeschnitten
	new_data = data[int(rate * start):]

	# ab 'end' abschneiden
	new_data = new_data[:int(rate * (end - start))]

	# auslesen des aktuellen Wortes
	os.makedirs(f"{data_dir}/{name}", mode=0o755, exist_ok=True)
	scipy.io.wavfile.write(f"{data_dir}/{name}/{name}_{ii:06d}.wav", rate, new_data)

	print('current iteration ' + str(ii))
	ii += 1

print('finished creating files')