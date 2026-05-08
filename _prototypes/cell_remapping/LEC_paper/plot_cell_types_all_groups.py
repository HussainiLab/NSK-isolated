import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon, ttest_rel, ttest_ind
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ColorConverter
import re
import matplotlib.gridspec as gridspec


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from library.study_space import Session, Study, SpatialSpikeTrain2D
from x_io.rw.axona.batch_read import make_study
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name_lec
from _prototypes.cell_remapping.src.masks import make_object_ratemap #, check_disk_arena
from library.maps.map_utils import _interpolate_matrix, disk_mask


def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = np.nan
    return  masked_rate_map.data


def _check_single_format(filename, format, fxn):
    if re.match(str(format), str(filename)) is not None:
        return fxn(filename)

def _find_subdir(folder_list,aid,date):
    """
    Finds the subdirectory of a given animal and date
    """
    date = str(date)
    aid = str(aid)
    matched_folders = []
    for folder in folder_list:
        if date in str(folder) and aid in str(folder):
            matched_folders.append(folder)
            # if aid in str(folder.split('/')[-2]):
            #     return folder

    # check folder has a file with .set extension
    for folder in matched_folders:
        for fle in os.listdir(folder):
            if '.set' in fle:
                return folder 


def main(dict_path, output_folder_path, folder_list, settings_dict, target_group='B6', target_animals=None):
    """
    Process cell types, generate plots, and compute average ratemaps and waveforms.
    
    Args:
        target_group: Group or list of groups to focus on (e.g., "B6" or ["B6", "ANT", "NON"])
        target_animals: List of animal names to include across target groups
    """
    if isinstance(target_group, str):
        target_groups = [target_group]
    else:
        target_groups = list(target_group)

    target_groups = [str(group_name) for group_name in target_groups]
    target_group_set = set(target_groups)

    if target_animals is None:
        target_animals = ["B6-1M", "B6-2M", "B6-LEC1", "B6-LEC2"]
    target_animals = [str(animal_name) for animal_name in target_animals]
    target_animals_set = set(target_animals)
    
    # Track missing subdirectories
    missing_subdir_records = []
    
    ########################## RATEMAP STORAGE ###################
    """
    ratemap_storage structure:
    {
        'overall': {
            'ANT': {'ratemap': np.array(32x32), 'count': int}  # All cells across all sessions/angles
        },
        'by_angle': {
            'ANT_0': {'ratemap': np.array(32x32), 'count': int},    # Cells at 0° object angle
            'ANT_90': {'ratemap': np.array(32x32), 'count': int},   # Cells at 90° object angle
            'ANT_180': {'ratemap': np.array(32x32), 'count': int},  # Cells at 180° object angle
            'ANT_270': {'ratemap': np.array(32x32), 'count': int},  # Cells at 270° object angle
            'ANT_NO': {'ratemap': np.array(32x32), 'count': int}    # Cells with no object
        },
        'by_session': {
            'ANT_session_1': {'ratemap': np.array(32x32), 'count': int},  # Cells in session 1
            'ANT_session_2': {'ratemap': np.array(32x32), 'count': int},  # Cells in session 2
            ...  # sessions 3-7
        },
        'by_celltype': {
            'ANT_object': {'ratemap': np.array(32x32), 'count': int},      # Object cells only
            'ANT_trace': {'ratemap': np.array(32x32), 'count': int},       # Trace cells only
            'ANT_unassigned': {'ratemap': np.array(32x32), 'count': int}   # Unassigned cells
        },
        'by_celltype_session': {
            'ANT_object_session_1': {'ratemap': np.array(32x32), 'count': int},  # Object cells in session 1
            'ANT_trace_session_1': {'ratemap': np.array(32x32), 'count': int},   # Trace cells in session 1
            ...  # All combinations of celltype × session
        }
    }
    
    Each 'ratemap' accumulates summed rate maps, 'count' tracks number of cells averaged.
    Final average = ratemap / count
    """
    ratemap_storage = {
        'overall': {},
        'by_angle': {},
        'by_session': {},
        'by_celltype': {},
        'by_celltype_session': {},
        'by_animal': {},
        'by_celltype_animal': {}
    }
    
    # Initialize ratemap storage for all target groups
    for group_name in target_groups:
        ratemap_storage['overall'][group_name] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}

        for angle in ['0', '90', '180', '270', 'NO']:
            ratemap_storage['by_angle'][f'{group_name}_{angle}'] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}

        for session_num in range(1, 8):
            ratemap_storage['by_session'][f'{group_name}_session_{session_num}'] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}

        for celltype in ['object', 'trace', 'unassigned']:
            ratemap_storage['by_celltype'][f'{group_name}_{celltype}'] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}

        for celltype in ['object', 'trace', 'unassigned']:
            for session_num in range(1, 8):
                key = f'{group_name}_{celltype}_session_{session_num}'
                ratemap_storage['by_celltype_session'][key] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}

    #initialize by animal ratemaps
    for animal in target_animals:
        ratemap_storage['by_animal'][animal] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}


    # initialize by animal and cell type ratemaps
    for celltype in ['object', 'trace', 'unassigned']:
        for animal in target_animals:
            key = f'{animal}_{celltype}'
            ratemap_storage['by_celltype_animal'][key] = {'ratemap': np.zeros((32, 32), dtype=float), 'count': 0}
    
    ########################## WAVEFORM STORAGE ###################
    """
    waveform_storage structure:
    {
        'ANT': {                                    # Group-level aggregation
            0: {                                    # Channel 0 (tetrode wire 1)
                'waveforms': np.array(200x50) or None,  # Accumulated waveforms (200 samples × 50 time points)
                'counts': int,                          # Number of times accumulated
                'dist': [float, ...]                    # Flattened distribution of all waveform values
            },
            1: { ... },  # Channel 1
            2: { ... },  # Channel 2
            3: { ... }   # Channel 3
        },
        'ANT-119a-6': {                            # Individual animal aggregation
            0: {'waveforms': np.array(200x50) or None, 'counts': int, 'dist': [...]},
            1: { ... },
            2: { ... },
            3: { ... }
        },
        'ANT-120-4': { ... },  # More individual animals
        ...
    }
    
    Each time waveforms are encountered:
    - 'waveforms': accumulated by summing (average = waveforms / counts)
    - 'counts': incremented by 1
    - 'dist': extended with all individual sample values for distribution analysis
    """
    # waveform_storage = {}
    # waveform_storage[target_group] = {}
    # for ch in range(4):
    #     waveform_storage[target_group][ch] = {'waveforms': None, 'counts': 0, 'dist': []}
    
    # for animal in target_animals:
    #     waveform_storage[animal] = {}
    #     for ch in range(4):
    #         waveform_storage[animal][ch] = {'waveforms': None, 'counts': 0, 'dist': []}
    
    ctype_dict = pd.read_pickle(dict_path)
    animal_group_map = {}
    for ctype_name in ctype_dict:
        for group_name in ctype_dict[ctype_name]:
            for animal_entry in ctype_dict[ctype_name][group_name]:
                animal_group = str(animal_entry[0])
                animal_id = str(animal_entry[1])
                if animal_group in target_group_set and animal_id in target_animals_set and animal_id not in animal_group_map:
                    animal_group_map[animal_id] = animal_group
    failed = []
    # for ctype in ['trace']:
    for ctype in ctype_dict:
        print(ctype)

        output_path = output_folder_path + '/' + str(ctype)
        
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for ctype_group in ctype_dict[ctype]:
            print(ctype_group)

            prev_aid = None 
            prev_date = None
            prev_tetrode = None 
            prev_cell_id = None

            animal_cell_list =  ctype_dict[ctype][ctype_group]
            
            for animal in animal_cell_list:
                # print(animal)
                animal_group = str(animal[0])
                aid = str(animal[1])
                if animal_group not in target_group_set or aid not in target_animals_set:
                    continue
                date = animal[3]
                tetrode = animal[4]
                cell_id = animal[5]

                save_path = output_path + '/' + str(aid) + '_' + str(date) + '_' + str(tetrode) + '_' + str(cell_id)
                

                settings_dict['single_tet'] = int(tetrode)
                settings_dict['allowed_sessions'] = None
                
                if True:
                    if aid == prev_aid and date == prev_date:
                        assert tetrode != prev_tetrode or cell_id != prev_cell_id, 'Duplicate cell found'
                    else:
                        subdir = _find_subdir(folder_list,aid,date)
                        print(f"subdir found: {subdir}")
 
                        if subdir is None and animal_group in target_group_set:
                            print(f"could not find subdir for {animal_group} animal {aid} on date {date}")
                            missing_subdir_records.append({
                                'animal_group': animal_group,
                                'animal_id': aid,
                                'date': date,
                                'tetrode': tetrode,
                                'cell_id': cell_id,
                                'expected_search_key': f"{aid}_{date}",
                                'folder_count_searched': len(folder_list)
                            })
                            continue

                        if subdir is not None:
                            print(subdir)
                            sub_study = make_study(subdir,settings_dict=settings_dict)
                            sub_study.make_animals()  
                            print('made study')    
                            print( sub_study.animals)

                            animal_obj = sub_study.animals[0]

                            ses_ratemaps = []
                            # ses_ratemap_objs = []
                            ses_ratemaps_raw = []
                            ses_ratemap_cells = []
                            angles = []

                            for ses_id in animal_obj.sessions:
                                print(ses_id)
                                session = animal_obj.sessions[ses_id]
                                pos_obj = session.get_position_data()['position']

                                tet_path = session.session_metadata.file_paths['tet']
                                fname = tet_path.split('/')[-1].split('.')[0]

                                if settings_dict['disk_arena']: 
                                    cylinder = True
                                else:
                                    cylinder, _ = check_disk_arena(fname)
                                    if not cylinder:
                                        print('Not cylinder for {}'.format(fname))
                                        
                                if settings_dict['naming_type'] == 'LEC':
                                    name_group, name = extract_name_lec(fname)
                                    formats = LEC_naming_format[name_group][name][settings_dict['type']]

                                for format in list(formats.keys()):
                                    checked = _check_single_format(fname, format, formats[format])
                                    if checked is not None:
                                        break
                                    else:
                                        continue
                                
                                angle, depth, name, date = checked
                                angles.append(angle)

                                for cell in session.get_cell_data()['cell_ensemble'].cells:  
                                    if int(cell.cluster.cluster_label) == int(cell_id):
                                        print(cell_id)

                                        spatial_spike_train = session.make_class(SpatialSpikeTrain2D, 
                                    {   'cell': cell, 'position': pos_obj, 'speed_bounds': (settings_dict['speed_lowerbound'], settings_dict['speed_upperbound'])})   

                        
                                        rate_obj = spatial_spike_train.get_map('rate')
                                        rate_map, rate_map_raw = rate_obj.get_rate_map(new_size=settings_dict['ratemap_dims'][0])

                                        ################ ACCUMULATE RATEMAPS ################
                                        if animal_group in target_group_set:
                                            # Overall ratemap
                                            ratemap_storage['overall'][animal_group]['ratemap'] += rate_map
                                            ratemap_storage['overall'][animal_group]['count'] += 1
                                            
                                            # By angle
                                            angle_key = f'{animal_group}_{angle}'
                                            if angle_key in ratemap_storage['by_angle']:
                                                ratemap_storage['by_angle'][angle_key]['ratemap'] += rate_map
                                                ratemap_storage['by_angle'][angle_key]['count'] += 1
                                            
                                            # By session
                                            session_key = f'{animal_group}_{ses_id}'
                                            if session_key in ratemap_storage['by_session']:
                                                ratemap_storage['by_session'][session_key]['ratemap'] += rate_map
                                                ratemap_storage['by_session'][session_key]['count'] += 1
                                            
                                            # By celltype
                                            celltype_key = f'{animal_group}_{ctype}'
                                            if celltype_key in ratemap_storage['by_celltype']:
                                                ratemap_storage['by_celltype'][celltype_key]['ratemap'] += rate_map
                                                ratemap_storage['by_celltype'][celltype_key]['count'] += 1
                                            
                                            # By celltype and session
                                            combined_key = f'{animal_group}_{ctype}_{ses_id}'
                                            if combined_key in ratemap_storage['by_celltype_session']:
                                                ratemap_storage['by_celltype_session'][combined_key]['ratemap'] += rate_map
                                                ratemap_storage['by_celltype_session'][combined_key]['count'] += 1

                                            # by animal
                                            curr_animal = animal[1]
                                            if curr_animal in ratemap_storage['by_animal']:
                                                ratemap_storage['by_animal'][curr_animal]['ratemap'] += rate_map
                                                ratemap_storage['by_animal'][curr_animal]['count'] += 1

                                            # by animal and cell type
                                            combined_animal_key = f'{curr_animal}_{ctype}'
                                            if combined_animal_key in ratemap_storage['by_celltype_animal']:
                                                ratemap_storage['by_celltype_animal'][combined_animal_key]['ratemap'] += rate_map
                                                ratemap_storage['by_celltype_animal'][combined_animal_key]['count'] += 1

                                        if cylinder:
                                            curr = flat_disk_mask(rate_map)
                                            
                                        ses_ratemaps.append(rate_map) 
                                        ses_ratemaps_raw.append(rate_map_raw)
                                        # ses_ratemap_objs.append(rate_obj)
                                        ses_ratemap_cells.append(cell)

                            fig = plt.figure(figsize=(4 * len(ses_ratemaps),4))
                            gs_main = gridspec.GridSpec(1, len(ses_ratemaps))
                            

                            for i, ratemap in enumerate(ses_ratemaps):
                                angle = angles[i]
                                print('plotting ratemap: ' + str(i))
                                gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[i], height_ratios=[12,2]) 

                                # ax = fig.add_subplot(1,len(ses_ratemaps),i+1)
                                ax = fig.add_subplot(gs_sub[0])
                                img = ax.imshow(ratemap, cmap='jet', aspect='equal')

                                # firing_rate = np.sum(ses_ratemaps_raw[i][~np.isnan(ses_ratemaps_raw[i])] / len(ses_ratemaps_raw[i][~np.isnan(ses_ratemaps_raw[i])]))
                                firing_rate = len(ses_ratemap_cells[i].event_times) / ses_ratemap_cells[i].event_times[-1] 
                                firing_rate = round(firing_rate,2)
                                # firing_rate2 = round(firing_rate2,2)
                                # angle_title = 'Angle: ' + str(angle) + ', Rate: ' + str(firing_rate) + ' Hz'
                                rate_title = str(firing_rate) + ' Hz' 
                                # + ', ' + str(firing_rate2) + ' Hz'
                                ax.set_title(rate_title, fontweight='bold')

                                if angle != 'NO':
                                    angle = int(angle)

                                    _, obj_loc = make_object_ratemap(angle, new_size=settings_dict['ratemap_dims'][0])
                                    
                                    if angle == 0:
                                        obj_loc['x'] += .5
                                        obj_loc['y'] += 2
                                    elif angle == 90:
                                        obj_loc['y'] += .5
                                        obj_loc['x'] -= 2
                                    elif angle == 180:
                                        obj_loc['x'] -= .5
                                        obj_loc['y'] -= 2
                                    elif angle == 270:
                                        obj_loc['y'] -= .5
                                        obj_loc['x'] += 2
                                    ax.plot(obj_loc['x'], obj_loc['y'], 'k', marker='o', markersize=20)

                                fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

                                ax2 = fig.add_subplot(gs_sub[1])
                                waveforms = ses_ratemap_cells[i].signal
                                for ch_id in range(4):
                                    if ch_id != len(waveforms):
                                        ch = waveforms[:,ch_id,:] # [spikes, channel, time], get all spikes at all time points for one channel
                                        idx = np.random.choice(len(ch), size=200)
                                        full_waves = ch[:,:] # get all spikes at all time points for this channel
                                        waves = ch[idx, :] # select 200 random spikes at all time points for this channel
                                        avg_wave = np.mean(ch, axis=0)

                                        ################ ACCUMULATE WAVEFORMS ################
                                        # animal_group = animal[0]
                                        # animal_name = animal[1]
                                        
                                        # if animal_group == target_group and ch_id < 4:
                                        #     ch_idx = ch_id
                                            
                                        #     # Add to group-level storage
                                        #     if waveform_storage[target_group][ch_idx]['waveforms'] is None:
                                        #         # waveform_storage[target_group][ch_idx]['waveforms'] = np.zeros_like(full_waves)
                                        #         waveform_storage[target_group][ch_idx]['waveforms'] = full_waves.copy()
                                        #     else:
                                        #         # waveform_storage[target_group][ch_idx]['waveforms'] += full_waves
                                        #         waveform_storage[target_group][ch_idx]['waveforms'] = np.vstack(( waveform_storage[target_group][ch_idx]['waveforms'],
                                        #                                                                         full_waves))
                                        #         waveform_storage[target_group][ch_idx]['counts'] += 1
                                        #         waveform_storage[target_group][ch_idx]['dist'].extend(full_waves.ravel())
                                                
                                        #     # Add to per-animal storage
                                        #     if animal_name in target_animals:
                                        #         if waveform_storage[animal_name][ch_idx]['waveforms'] is None:
                                        #             # waveform_storage[animal_name][ch_idx]['waveforms'] = np.zeros_like(full_waves)
                                        #             waveform_storage[animal_name][ch_idx]['waveforms'] = full_waves.copy()
                                        #         else:
                                        #             # waveform_storage[animal_name][ch_idx]['waveforms'] += full_waves
                                        #             waveform_storage[animal_name][ch_idx]['waveforms'] = np.vstack((waveform_storage[animal_name][ch_idx]['waveforms'],
                                        #                                                                            full_waves))
                                        #             waveform_storage[animal_name][ch_idx]['counts'] += 1
                                        #             waveform_storage[animal_name][ch_idx]['dist'].extend(full_waves.ravel())

                                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), ch[idx,:].T, c='grey')
                                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), avg_wave, c='k', lw=2)
                                        
                                        ax2.set_xlim([-25,200])
                                        ax2.spines['top'].set_visible(False)
                                        ax2.spines['bottom'].set_visible(False)
                                        ax2.spines['right'].set_visible(False)
                                        ax2.spines['left'].set_visible(False)

                                ax2.tick_params(
                                    axis='x',          # changes apply to the x-axis
                                    which='both',      # both major and minor ticks are affected
                                    bottom=False,      # ticks along the bottom edge are off
                                    top=False,         # ticks along the top edge are off
                                    labelbottom=False)
                        

                                ax2.tick_params(
                                    axis='y',          # changes apply to the x-axis
                                    which='both',      # both major and minor ticks are affected
                                    bottom=False,      # ticks along the bottom edge are off
                                    top=False,         # ticks along the top edge are off
                                    labelbottom=False)

                                ax2.set_xticks([])
                                ax2.set_yticks([])
                            
                            title = str(ctype) + ' cell - animal: ' + str(aid) + ', date: ' + str(date) + ', depth: ' + str(depth) + ', tetrode: ' + str(tetrode) + ', unit: ' + str(cell_id)

                            # fig.suptitle(str(ctype) + ' cell ' + str(animal), fontweight='bold')
                            fig.suptitle(title)
                            fig.tight_layout()

                            # plt.show()
                            # stop()
                            # save to output folder
                            fig.savefig(save_path , dpi=360)
                            plt.close()
                            print(save_path)  
                            # stop()

                            prev_date = date
                            prev_cell_id = cell_id
                            prev_aid = aid 
                            prev_tetrode = tetrode
                # except:
                #     failed.append(animal)

    ########################## SAVE RATEMAPS ###################
    print(f"Saving ratemaps...")
    avg_ratemap_folder = output_folder_path + '/avg_ratemap_npy'
    no_mask_folder = avg_ratemap_folder + '/no_mask'
    if not os.path.isdir(avg_ratemap_folder):
        os.mkdir(avg_ratemap_folder)
    if not os.path.isdir(no_mask_folder):
        os.mkdir(no_mask_folder)
    
    ratemap_keys = []
    
    # Save all ratemaps using clean loop-based approach
    for key, data in ratemap_storage['overall'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))
    
    for key, data in ratemap_storage['by_angle'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))
    
    for key, data in ratemap_storage['by_session'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))
    
    for key, data in ratemap_storage['by_celltype'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))
    
    for key, data in ratemap_storage['by_celltype_session'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))

    #for animal-level ratemaps
    for key, data in ratemap_storage['by_animal'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))

    # for animal-cell type ratemaps
    for key, data in ratemap_storage['by_celltype_animal'].items():
        if data['count'] > 0:
            avg_ratemap = data['ratemap'] / data['count']
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{key}.npy')
            np.save(unmasked_path, avg_ratemap)
            masked_ratemap = flat_disk_mask(avg_ratemap)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{key}.npy')
            np.save(masked_path, masked_ratemap)
            print(f"Saved {masked_path}")
            ratemap_keys.append((key, data['count']))

    #for average group average created by averaging of animal rate maps
    for group_name in target_groups:
        overall_avg = np.zeros((32,32), dtype=float)
        num_animals = 0
        for animal_name, data in ratemap_storage['by_animal'].items():
            if data['count'] > 0 and animal_group_map.get(animal_name) == group_name:
                overall_avg += data['ratemap'] / data['count'] #add the average animal rate map to overall average
                num_animals += 1

        if num_animals > 0:
            overall_avg = overall_avg / num_animals #divide by number of animals to get average of animal averages
            unmasked_path = os.path.join(no_mask_folder, f'average_ratemap_{group_name}_animal_avg.npy')
            np.save(unmasked_path, overall_avg)
            masked_ratemap = flat_disk_mask(overall_avg)
            masked_path = os.path.join(avg_ratemap_folder, f'average_ratemap_{group_name}_animal_avg.npy')
            np.save(masked_path, masked_ratemap)
    
    ########################## SAVE WAVEFORMS ###################
    # print(f"Saving waveforms...")
    # avg_waveform_folder = output_folder_path + '/avg_waveform_npy'
    # if not os.path.isdir(avg_waveform_folder):
    #     os.mkdir(avg_waveform_folder)
    
    # waveform_keys = []
    
    # for entity_id in [target_group] + target_animals:
    #     for ch_idx in range(4):
    #         data = waveform_storage[entity_id][ch_idx]
            
    #         # Save averaged waveforms
    #         if data['counts'] > 0:
    #             # avg_waveforms = data['waveforms'] / data['counts']
    #             avg_waveforms = np.mean(data['waveforms'], axis=0)  # Average across all accumulated waveforms
    #             npy_path = os.path.join(avg_waveform_folder, f'{entity_id}_ch{ch_idx + 1}_waveforms.npy')
    #             np.save(npy_path, avg_waveforms)
    #             print(f"Saved {npy_path}")
            
    #         # Save distributions
    #         if data['dist']:
    #             dist_path = os.path.join(avg_waveform_folder, f'{entity_id}_ch{ch_idx + 1}_dist.npy')
    #             np.save(dist_path, np.array(data['dist']))
    #             print(f"Saved {dist_path}")
            
    #         # Track counts
    #         waveform_keys.append((f'{entity_id}_ch{ch_idx + 1}', data['counts']))
    
    ########################## SAVE CSV COUNTS ###################
    # Write ratemap counts to CSV
    ratemap_counts_data = {'Ratemap Name': [], 'Ratemap Count': []}
    for key, count in ratemap_keys:
        ratemap_counts_data['Ratemap Name'].append(f'{key}_ratemap_count')
        ratemap_counts_data['Ratemap Count'].append(count)
    
    print(f"Writing ratemap counts CSV...")
    ratemap_counts_df = pd.DataFrame(ratemap_counts_data)
    csv_group_tag = '_'.join(target_groups)
    ratemap_counts_df.to_csv(output_folder_path + f'/{csv_group_tag}_ratemap_counts.csv', index=False)
    
    # # Write waveform counts to CSV
    # waveform_counts_data = {'Waveform Name': [], 'Waveform Count': []}
    # for key, count in waveform_keys:
    #     waveform_counts_data['Waveform Name'].append(f'{key}_waveform_count')
    #     waveform_counts_data['Waveform Count'].append(count)
    
    # print(f"Writing waveform counts CSV...")
    # waveform_counts_df = pd.DataFrame(waveform_counts_data)
    # waveform_counts_df.to_csv(output_folder_path + f'/{target_group}_waveform_counts.csv', index=False)

    # Write missing subdirectory records to CSV
    if missing_subdir_records:
        print(f"Writing missing subdirectory records CSV...")
        missing_records_df = pd.DataFrame(missing_subdir_records)
        missing_records_df.to_csv(output_folder_path + f'/{csv_group_tag}_missing_subdir_records.csv', index=False)
        print(f"Found {len(missing_subdir_records)} missing subdirectories")
    else:
        print(f"No missing subdirectories found for target groups: {target_groups}")

    return failed


if __name__ == '__main__':

    STUDY_SETTINGS = {

        'ppm': 485,  # EDIT HERE

        'smoothing_factor': 3, # EDIT HERE

        'useMatchedCut': False,  # EDIT HERE
    }

    # Switch devices to True/False based on what is used in the acquisition (to be extended for more devices in future)
    device_settings = {'axona_led_tracker': True, 'implant': True} 
    # Make sure implant metadata is correct, change if not, AT THE MINIMUM leave implant_type: tetrode
    implant_settings = {'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}
    # WE ASSUME DEVICE AND IMPLANT SETTINGS ARE CONSISTENCE ACROSS SESSIONS
    # Set channel count + add device/implant settings
    SESSION_SETTINGS = {
        'channel_count': 4, # EDIT HERE, default is 4, you can change to other number but code will check how many tetrode files are present and set that to channel copunt regardless
        'devices': device_settings, # EDIT HERE
        'implant': implant_settings, # EDIT HERE
    }
    STUDY_SETTINGS['session'] = SESSION_SETTINGS
    settings_dict = STUDY_SETTINGS

    settings_dict['speed_lowerbound'] = 0 
    settings_dict['speed_upperbound'] = 100
    settings_dict['ratemap_dims'] = (32,32)
    settings_dict['disk_arena'] = True
    settings_dict['naming_type'] = 'LEC'
    settings_dict['type'] = 'object'

    #julian addeds
    settings_dict['arena_size'] = None

    folder_path = r"e:\Julian\Remove_low_spike_units\ALL_SORTING_thres8_output_no_odor\remove_low_spikes"

    # output_folder_path = r"e:\Julian\ANDREW_CSV\plot_cell_types_B6_animals\output"
    output_folder_path = r"e:\Julian\plot_cell_types\duration_correct\threshold_8\output_all_groups"

    dict_path = r"e:\Julian\plot_cell_types\duration_correct\threshold_8\input\ses123_identifier_dict_keep_swapped.pkl"

    # list all folders in folder_path at any level  
    folder_list = [x[0] for x in os.walk(folder_path)]
    folder_list = [x.replace('\\', '/') for x in folder_list]

    # Specify target groups and animals
    target_group = ['B6', 'ANT', 'NON']
    target_animals = ['B6-1M', 'B6-2M', 'B6-LEC1', 'B6-LEC2',
                      'NON-73-6', 'NON-88-1', 'NON-INT-01', 'NON-INT-02', 'NON-INT-03',
                      'ANT-119a-6', 'ANT-120-4', 'ANT-133a-4', 'ANT-135a-7', 'ANT-140-4']
    
    failed = main(dict_path, output_folder_path, folder_list, settings_dict, target_group, target_animals)

    # print('Failed IDs')

    # print(failed)