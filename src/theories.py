""" Kernels are represented as dictionary, with the keys representing the
different emotions and the values the corresponding hypothesized AU-configurations.
Note that the values can be dictionary themselves, representing different *possible*
configurations (basically, when a theory states: "anger can be either [AUx, AUx, AUx] *or*
[AUx, AUx, AUx, AUx]"). """

THEORIES = dict(
    Darwin=dict(
        happiness=['AU06L', 'AU06R', 'AU12L', 'AU12R'],
        sadness=['AU01', 'AU15'],
        surprise={
            0: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26']
        },
        fear=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU20L', 'AU20R'],
        anger=['AU04', 'AU05', 'AU24', 'AU38'],
        disgust={
            0: ['AU10L', 'AU10R', 'AU16', 'AU22', 'AU25'],
            1: ['AU10L', 'AU10R', 'AU16', 'AU22', 'AU26']
        }
    ),
    Matsumoto2008=dict(
        happiness=['AU06L', 'AU06R' 'AU12L', 'AU12R'],
        sadness={
            0: ['AU01', 'AU15'],
            1: ['AU04'],
            2: ['AU04', 'AU01', 'AU15'],
            3: ['AU17'],
            4: ['AU17', 'AU01', 'AU15'],
            5: ['AU17', 'AU01', 'AU15', 'AU04']
        },
        surprise={
            0: ['AU01', 'AU2L', 'AU2R', 'AU05', 'AU25'],
            1: ['AU01', 'AU2L', 'AU2R', 'AU05', 'AU26']
        },
        fear={
            0: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20'],
            1: ['AU25'],
            2: ['AU26']
        },
        anger={
            0: ['AU04', 'AU05', 'AU22', 'AU23', 'AU24'],
            1: ['AU04', 'AU07L', 'AU07R', 'AU22', 'AU23', 'AU24'] 
        },
        disgust={
            0: ['AU09'],
            1: ['AU10L', 'AU10R'],
            2: ['AU25'],
            3: ['AU26'],
            4: ['AU09', 'AU25'],
            5: ['AU10L', 'AU10R', 'AU26']
        }
    ),
    Keltner2019=dict(
        happiness=['AU06L', 'AU06R', 'AU07', 'AU12L', 'AU12R' 'AU25', 'AU26'],
        sadness=['AU01', 'AU04', 'AU06L', 'AU06R', 'AU15', 'AU17'],
        surprise=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25', 'AU26'],
        fear=['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU07L', 'AU07R', 'AU20', 'AU25'],
        anger=['AU04', 'AU05', 'AU17', 'AU23', 'AU24'],
        disgust=['AU07', 'AU09', 'AU25', 'AU26']  # misses AU19, tongue show
    ),
    Cordaro2008ref=dict(
        happiness=['AU06L', 'AU06R', 'AU12L', 'AU12R'],
        sadness=['AU01', 'AU04', 'AU05'],
        surprise=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26'],
        fear=['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU07L', 'AU07R', 'AU20', 'AU26'],
        anger=['AU04', 'AU05', 'AU07L', 'AU07R', 'AU23'],
        disgust=['AU09', 'AU15', 'AU16']
    ),
    Cordaro2008IPC=dict(
        happiness={
            0: ['AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU12L', 'AU12R', 'AU16', 'AU25', 'AU26'],
            1: ['AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU12L', 'AU12R', 'AU16', 'AU25', 'AU27'],
        },
        sadness=['AU04', 'AU43'],  # misses AU54 (head down)
        surprise=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25'],
        fear=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU07', 'AU25'],  # also "jaw"/"move back"
        anger=['AU04', 'AU07L', 'AU07R'],
        disgust=['AU04', 'AU06', 'AU07L', 'AU07R', 'AU09', 'AU10', 'AU25', 'AU26']  # also "jaw"
    )
)