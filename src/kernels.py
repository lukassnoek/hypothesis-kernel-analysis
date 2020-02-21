""" Kernels are represented as dictionary, with the keys representing the
different emotions and the values the corresponding hypothesized AU-configurations.
Note that the values can be dictionary themselves, representing different *possible*
configurations (basically, when a theory states: "anger can be either [AUx, AUx, AUx] *or*
[AUx, AUx, AUx, AUx]"). """
    
theory_kernels = dict(
    Darwin=dict(
        happiness=['AU6', 'AU12'],
        sadness=['AU1', 'AU15'],
        surprise={
            0: ['AU1', 'AU2', 'AU5', 'AU25'],
            1: ['AU1', 'AU2', 'AU5', 'AU26']
        },
        fear=['AU1', 'AU2', 'AU5', 'AU20'],
        anger=['AU4', 'AU5', 'AU24', 'AU38'],
        disgust={
            0: ['AU10Open', 'AU16Open', 'AU22', 'AU25'],
            1: ['AU10Open', 'AU16Open', 'AU22', 'AU26']
        }
    ),
    Matsumoto2008=dict(
        happiness=['AU6', 'AU12'],
        sadness={
            0: ['AU1', 'AU15'],
            1: ['AU4'],
            2: ['AU4', 'AU1', 'AU15'],
            3: ['AU17'],
            4: ['AU17', 'AU1', 'AU15'],
            5: ['AU17', 'AU1', 'AU15', 'AU4']
        },
        surprise={
            0: ['AU1', 'AU2', 'AU5', 'AU25'],
            1: ['AU1', 'AU2', 'AU5', 'AU26']
        },
        fear={
            0: ['AU1', 'AU2', 'AU4', 'AU5', 'AU20'],
            1: ['AU25'],
            2: ['AU26']
        },
        anger={
            0: ['AU4', 'AU5', 'AU22', 'AU23', 'AU24'],
            1: ['AU4', 'AU7', 'AU22', 'AU23', 'AU24'] 
        },
        disgust={
            0: ['AU9'],
            1: ['AU10Open'],
            2: ['AU25'],
            3: ['AU26'],
            4: ['AU9', 'AU25'],
            5: ['AU10Open', 'AU26']
        }
    ),
    Keltner2019=dict(
        happiness=['AU6', 'AU7', 'AU12', 'AU25', 'AU26'],
        sadness=['AU1', 'AU4', 'AU6', 'AU15', 'AU17'],
        surprise=['AU1', 'AU2', 'AU5', 'AU25', 'AU26'],
        fear=['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU25'],
        anger=['AU4', 'AU5', 'AU17', 'AU23', 'AU24'], 
        disgust=['AU7', 'AU9', 'AU25', 'AU26']  # misses AU19, tongue show
    ),
    Cordaro2008ref=dict(
        happiness=['AU6', 'AU12'],
        sadness=['AU1', 'AU4', 'AU5'],
        surprise=['AU1', 'AU2', 'AU5', 'AU26'],
        fear=['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU26'],
        anger=['AU4', 'AU5', 'AU7', 'AU23'],
        disgust=['AU9', 'AU15', 'AU16Open']
    ),
    Cordaro2008IPC=dict(
        happiness={
            0: ['AU6', 'AU7', 'AU12', 'AU16Open', 'AU25', 'AU26'],
            1: ['AU6', 'AU7', 'AU12', 'AU16Open', 'AU25', 'AU27i'],
        },
        sadness=['AU4', 'AU43'],  # misses AU54 (head down)
        surprise=['AU1', 'AU2', 'AU5', 'AU25'],
        fear=['AU1', 'AU2', 'AU5', 'AU7', 'AU25'],  # also "jaw"/"move back"
        anger=['AU4', 'AU7'],
        disgust=['AU4', 'AU6', 'AU7', 'AU9', 'AU10Open', 'AU25', 'AU26']  # also "jaw"
    )
)
