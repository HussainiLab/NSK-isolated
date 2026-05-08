import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from x_io.rw.axona.batch_read import make_study


def _init_storage_for_entity(storage, entity_id, channel_count, bin_edges=None):
	if entity_id not in storage:
		storage[entity_id] = {}
		# Define histogram bins for voltage distribution (adjust range as needed)
		if bin_edges is None:
			
			bin_edges = np.arange(-150, 150 + 1)
		
		for ch in range(channel_count):
			# Store running mean, running variance (M2), and count for incremental computation
			storage[entity_id][ch] = {
				'mean_waveform': None,  # Running mean of waveforms
				'M2': None,  # Running variance accumulator (for std if needed)
				'counts': 0,  # Total number of waveforms seen
				'histogram': np.zeros(len(bin_edges) - 1, dtype=np.int64),  # Histogram bins
				'bin_edges': bin_edges.copy()  # Bin edges for histogram
			}


def _accumulate_waveforms(storage, entity_id, waveforms, channel_count=4, sample_size=200):
	_init_storage_for_entity(storage, entity_id, channel_count)

	if waveforms is None:
		return
	if not hasattr(waveforms, 'ndim') or waveforms.ndim != 3:
		return

	spike_count, ch_count, timepoints = waveforms.shape
	if spike_count == 0 or ch_count == 0 or timepoints == 0:
		return

	for ch_idx in range(4):
		ch = waveforms[:, ch_idx, :]
		if len(ch) == 0:
			continue

		# Incremental mean update using Welford's algorithm
		for waveform in ch:
			n = storage[entity_id][ch_idx]['counts']
			
			if storage[entity_id][ch_idx]['mean_waveform'] is None:
				# First waveform - initialize
				storage[entity_id][ch_idx]['mean_waveform'] = waveform.copy()
				storage[entity_id][ch_idx]['M2'] = np.zeros_like(waveform)
			else:
				# Update running mean: new_mean = old_mean + (value - old_mean) / (n + 1)
				delta = waveform - storage[entity_id][ch_idx]['mean_waveform']
				storage[entity_id][ch_idx]['mean_waveform'] += delta / (n + 1)
				
				# Update M2 for variance (optional, if you need std later)
				delta2 = waveform - storage[entity_id][ch_idx]['mean_waveform']
				storage[entity_id][ch_idx]['M2'] += delta * delta2
			
			storage[entity_id][ch_idx]['counts'] += 1
			
			# Update histogram incrementally (bin the values without storing them)
			flat_values = waveform.ravel()
			bin_edges = storage[entity_id][ch_idx]['bin_edges']
			# Use digitize to find which bin each value belongs to, then update counts
			bin_indices = np.digitize(flat_values, bin_edges[1:-1])
			# Count values in each bin and add to histogram
			# for bin_idx in bin_indices:
			# 	if 0 <= bin_idx < len(storage[entity_id][ch_idx]['histogram']):
			# 		storage[entity_id][ch_idx]['histogram'][bin_idx] += 1
			hist = storage[entity_id][ch_idx]['histogram']
			hist += np.bincount(bin_indices, minlength=hist.size)

def collect_waveforms_from_study(study, storage, entity_prefix=None, channel_count=4, sample_size=200):
	if study.animals is None:
		study.make_animals()

	for animal in study.animals:
		animal_id = animal.animal_id.split('_tet')[0]
		# Use entity_prefix if provided (for group/animal naming)
		if entity_prefix:
			storage_key = entity_prefix
		else:
			storage_key = animal_id
		
		_init_storage_for_entity(storage, storage_key, channel_count)

		for session_key in animal.sessions:
			session = animal.sessions[session_key]

			cells = session.get_cell_data()['cell_ensemble'].cells
			for cell in cells:
				waveforms = cell.signal
				_accumulate_waveforms(storage, storage_key, waveforms, channel_count, sample_size)

def save_waveforms(storage, output_folder, channel_count=4):
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder, exist_ok=True)

	waveform_counts_data = {'Waveform Name': [], 'Waveform Count': []}

	for entity_id in storage:
		for ch_idx in range(channel_count):
			data = storage[entity_id][ch_idx]

			if data['counts'] > 0:
				# The mean is already computed incrementally!
				avg_waveforms = data['mean_waveform']
				print(f"DEBUG {entity_id}_ch{ch_idx + 1}: avg_waveforms shape={avg_waveforms.shape}, values={avg_waveforms[:5]}")
				npy_path = os.path.join(output_folder, f'{entity_id}_ch{ch_idx + 1}_waveforms.npy')
				print(f"Saving {entity_id}_ch{ch_idx + 1}: shape={avg_waveforms.shape}, counts={data['counts']}")
				np.save(npy_path, avg_waveforms)

			if np.sum(data['histogram']) > 0:
				# Save histogram data (bin edges and counts)
				hist_data = {
					'counts': data['histogram'],
					'bin_edges': data['bin_edges']
				}
				
				dist_path = os.path.join(output_folder, f'{entity_id}_ch{ch_idx + 1}_histogram.npy')
				total_values = np.sum(data['histogram'])
				print(f"Saving histogram {entity_id}_ch{ch_idx + 1}: {len(data['histogram'])} bins, {total_values} total values")
				np.save(dist_path, hist_data)

			waveform_counts_data['Waveform Name'].append(f'{entity_id}_ch{ch_idx + 1}_waveform_count')
			waveform_counts_data['Waveform Count'].append(data['counts'])

	waveform_counts_df = pd.DataFrame(waveform_counts_data)
	waveform_counts_df.to_csv(os.path.join(output_folder, 'waveform_counts.csv'), index=False)


def main():
	# Set data directory, group, and animals
	data_dir = r"z:\Users\Andrew\LEC\all_LEC_data\all_animals\all_completed_copies (made by julian, for running avg waveform)\B6-1M"
	group = None #change to None
	animals = ['B6-1M']
	
	# Validate that either group or animals are provided
	if not group and not animals:
		print("Error: Either group or animals must be specified")
		sys.exit(1)
	
	# Validate the directory exists
	if not os.path.isdir(data_dir):
		print(f"Error: Directory '{data_dir}' does not exist")
		sys.exit(1)

	animal_template = {
		'animal_id': '001',
		'species': 'mouse',
		'sex': 'F',
		'age': 1,
		'weight': 1,
		'genotype': 'type',
		'animal_notes': 'notes'
	}
	devices = {'axona_led_tracker': True, 'implant': True}
	implant = {
		'implant_id': '001',
		'implant_type': 'tetrode',
		'implant_geometry': 'square',
		'wire_length': 25,
		'wire_length_units': 'um',
		'implant_units': 'uV'
	}

	session_settings = {'channel_count': 4, 'animal': animal_template, 'devices': devices, 'implant': implant}
	settings = {'ppm': 485, 'session': session_settings, 'smoothing_factor': 3, 'useMatchedCut': False}
	settings['disk_arena'] = True
	settings['naming_type'] = 'LEC'
	settings['arena_size'] = None
	settings['speed_lowerbound'] = 0
	settings['speed_upperbound'] = 100
	settings['end_cell'] = None
	settings['start_cell'] = None
	settings['saveData'] = True

	start_time = time.time()

	storage = {}

	# If group is provided, process group data
	if group:
		_init_storage_for_entity(storage, group, session_settings['channel_count'])

	# Initialize storage for each animal
	for animal_id in animals:
		_init_storage_for_entity(storage, animal_id, session_settings['channel_count'])

	subdirs = np.sort([f.path for f in os.scandir(data_dir) if f.is_dir()])
	for subdir in subdirs:
		try:
			study = make_study(subdir, settings_dict=settings)
			study.make_animals()
			
			# Process for group if specified
			if group:
				collect_waveforms_from_study(
					study,
					storage,
					entity_prefix=group,
					channel_count=session_settings['channel_count'],
					sample_size=200
				)
			
			# Process for each animal if specified
			for animal_id in animals:
				collect_waveforms_from_study(
					study,
					storage,
					entity_prefix=animal_id,
					channel_count=session_settings['channel_count'],
					sample_size=200
				)
		except Exception:
			print(traceback.format_exc())
			print('DID NOT WORK FOR DIRECTORY ' + str(subdir))

	output_folder = os.path.join(data_dir, 'avg_waveforms')
	save_waveforms(storage, output_folder, channel_count=session_settings['channel_count'])

	elapsed = time.time() - start_time
	print(f"Completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
	main()
