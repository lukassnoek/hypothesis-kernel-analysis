""" Kernels are represented as dictionary, with the keys representing the
different emotions and the values the corresponding hypothesized AU-configurations.
Note that the values can be dictionary themselves, representing different *possible*
configurations (basically, when a theory states: "anger can be either [AUx, AUx, AUx] *or*
[AUx, AUx, AUx, AUx]"). """

MAPPINGS = dict(
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
        happiness=['AU06L', 'AU06R', 'AU12L', 'AU12R'],
        sadness={
            0: ['AU01', 'AU15'],
            1: ['AU01', 'AU15', 'AU04'],
            2: ['AU01', 'AU15', 'AU17'],
            3: ['AU01', 'AU15', 'AU04', 'AU17']
        },
        surprise={
            0: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26']
        },
        fear={
            0: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20', 'AU25'],
            2: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20', 'AU26'],
        },
        anger={
            0: ['AU04', 'AU05', 'AU22', 'AU23', 'AU24'],
            1: ['AU04', 'AU07L', 'AU07R', 'AU22', 'AU23', 'AU24'] 
        },
        disgust={
            0: ['AU09'],
            1: ['AU10L', 'AU10R'],
            2: ['AU09', 'AU25'],
            3: ['AU09', 'AU26'],
            4: ['AU10L', 'AU10R', 'AU25'],
            5: ['AU10L', 'AU10R', 'AU26']
        }
    ),
    Keltner2019=dict(
        happiness=['AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU12L', 'AU12R', 'AU25', 'AU26'],
        sadness=['AU01', 'AU04', 'AU06L', 'AU06R', 'AU15', 'AU17'],
        surprise=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25', 'AU26'],
        fear=['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU07L', 'AU07R', 'AU20', 'AU25'],
        anger=['AU04', 'AU05', 'AU17', 'AU23', 'AU24'],
        disgust=['AU07L', 'AU07R', 'AU09', 'AU25', 'AU26']  # misses AU19, tongue show
    ),
    Cordaro2018ref=dict(
        happiness=['AU06L', 'AU06R', 'AU12L', 'AU12R'],
        sadness=['AU01', 'AU04', 'AU05'],
        surprise=['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26'],
        fear=['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU07L', 'AU07R', 'AU20', 'AU26'],
        anger=['AU04', 'AU05', 'AU07L', 'AU07R', 'AU23'],
        disgust=['AU09', 'AU15', 'AU16']
    ),
    Cordaro2018IPC=dict(
        happiness={
            0: ['AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU12L', 'AU12R', 'AU16', 'AU25', 'AU26'],
            1: ['AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU12L', 'AU12R', 'AU16', 'AU25', 'AU27'],
        },
        sadness=['AU04', 'AU43'],  # misses AU54 (head down)
        surprise={
            0: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25', 'AU26'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25', 'AU27']
        },
        fear={
            0: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU07L', 'AU07R', 'AU25', 'AU26'],  # also "jaw"/"move back"
            1: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU07L', 'AU07R', 'AU25', 'AU27']
        },
        anger=['AU04', 'AU07L', 'AU07R'],
        disgust={
            0: ['AU04', 'AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU09', 'AU10L', 'AU10R', 'AU25', 'AU26'],  # also "jaw"
            1: ['AU04', 'AU06L', 'AU06R', 'AU07L', 'AU07R', 'AU09', 'AU10L', 'AU10R', 'AU25', 'AU27']
        }
    ),
    #JackAndSchyns=dict(
    #    happiness=,
    #    sadness=,
    #    surprise=,
    #    fear=
    #    anger=
    #    disgust=
    #)
    #
    # from https://www.frontiersin.org/articles/10.3389/fpsyg.2020.00920/full
    Ekman=dict(
        happiness={
            0: ['AU12L', 'AU12R'],
            1: ['AU12L', 'AU12R', 'AU06L', 'AU06R']
        },
        sadness={
            0: ['AU01', 'AU04'],
            1: ['AU01', 'AU04', 'AU11L', 'AU11R'],
            2: ['AU01', 'AU04', 'AU15'],
            3: ['AU01', 'AU04', 'AU15', 'AU17'],
            4: ['AU06L', 'AU06R', 'AU15'],
            5: ['AU11L', 'AU11R', 'AU17'],
            6: ['AU01']
        },
        surprise={
            0: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU27'],
            2: ['AU01', 'AU02L', 'AU02R', 'AU05'],
            3: ['AU01', 'AU02L', 'AU02R', 'AU26'],
            4: ['AU01', 'AU02L', 'AU02R', 'AU27'],
            5: ['AU05', 'AU26'],
            6: ['AU05', 'AU27']
        },
        fear={
            0: ['AU01', 'AU02L', 'AU02R', 'AU04'],
            1: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20', 'AU25'],
            2: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20', 'AU26'],
            3: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU20', 'AU27'],
            4: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU25'],
            5: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU26'],
            6: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05', 'AU27'],
            7: ['AU01', 'AU02L', 'AU02R', 'AU04', 'AU05'],
            8: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU25'],
            9: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU26'],
            10: ['AU01', 'AU02L', 'AU02R', 'AU05', 'AU27'],
            11: ['AU05', 'AU20', 'AU25'],
            12: ['AU05', 'AU20', 'AU26'],
            13: ['AU05', 'AU20', 'AU27'],
            14: ['AU05', 'AU20'],
            15: ['AU20']
        },
        anger={
            0: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU10L', 'AU10R', 'AU11L', 'AU11R', 'AU22', 'AU23', 'AU25'],
            1: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU10L', 'AU10R', 'AU11L', 'AU11R', 'AU22', 'AU23', 'AU26'],
            2: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU10L', 'AU10R', 'AU11L', 'AU11R', 'AU23', 'AU25'],
            3: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU10L', 'AU10R', 'AU11L', 'AU11R', 'AU23', 'AU26'],
            4: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU17', 'AU23'],
            5: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU17', 'AU24'],
            6: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU24'],
            7: ['AU04', 'AU05', 'AU07L', 'AU07R', 'AU24'],
            8: ['AU04', 'AU05'],
            9: ['AU04', 'AU07L', 'AU07R'],
            10: ['AU17', 'AU24']
        },
        disgust={
            0: ['AU09', 'AU17'],
            1: ['AU10L', 'AU10R', 'AU17'],
            2: ['AU09', 'AU16', 'AU25'],
            3: ['AU10L', 'AU10R', 'AU16', 'AU25'],
            4: ['AU09', 'AU16', 'AU26'],
            5: ['AU10L', 'AU10R', 'AU16', 'AU26'],
            6: ['AU09'],
            7: ['AU10L', 'AU10R']
        }
    )
)