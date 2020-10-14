import numpy as np
PARAM_NAMES = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def simulate_configs(n_mapp, max_config):
    MAPPINGS_configs = dict()
    list_configs = np.empty([0])
    for i in range(n_mapp):
        n_configs = np.random.randint(1, max_config)    
        configs_per_mapping = dict()
        for emotion in EMOTIONS:
        
            configs_per_emotion = dict()
            for config in range(n_configs):
                n_aus = np.random.choice(np.arange(1,12), p=[0.04040404,0.15151515,0.17171717,0.14141414,
                0.19191919,0.12121212,0.05050505,0.03030303,0.04040404,0.04040404, 0.02020202])
                configs_per_emotion[config] = list(np.random.choice(PARAM_NAMES, n_aus))
            
            configs_per_mapping[emotion] = configs_per_emotion
            
        MAPPINGS_configs[f'mapp_{i}'] = configs_per_mapping
        list_configs = np.append(list_configs, n_configs)
    
    return MAPPINGS_configs, list_configs

def simulate_aus(n_mapp, max_aus):
    MAPPINGS_aus = dict()
    list_aus = np.empty([0])
    for i in range(n_mapp):
        n_aus = np.random.randint(1, max_aus)
    
        configs_per_mapping = dict()
        for emotion in EMOTIONS:
            configs_per_mapping[emotion] = list(np.random.choice(PARAM_NAMES, n_aus))
            
        MAPPINGS_aus[f'mapp_{i}'] = configs_per_mapping
        list_aus = np.append(list_aus, n_aus)

    return MAPPINGS_aus, list_aus


if __name__ == '__main__':
    simulate_configs(1, 10)