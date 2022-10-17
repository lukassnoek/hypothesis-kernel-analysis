MAPPINGS = dict(
    Cunningham=dict(
        thinking=['AU04'],
        confused=['AU04', 'AU20L', 'AU20R']
    ),
    Ekman=dict(
        thinking=['AU04', 'AU01', 'AU02L', 'AU02R'],
        confused={
            0: ['AU04'],
            1: ['AU01', 'AU02L', 'AU02R']
        }
    ),
    Forsyth=dict(
        interested=['AU12L', 'AU12R', 'AU06L', 'AU06R', 'AU01', 'AU02L', 'AU02R'],
        bored=[]
    ),
    # Rozin=dict(
    #     thinking={
    #         0: ['AU01', 'AU02L', 'AU02R', 'AU07L', 'AU07R'],
    #         1: ['AU04', 'AU07L', 'AU07R']
    #     },
    #     confused={
    #         0: ['AU02L', 'AU02R', 'AU07L', 'AU07R'],
    #         1: ['AU02L'],
    #         2: ['AU02R']
    #     }
    # ),
    Kaliouby=dict(
        thinking=['AU23', 'AU24'],
        interested={
            0: ['AU26', 'AU12L', 'AU12R', 'AU01', 'AU02L', 'AU02R'],
            1: ['AU27', 'AU12L', 'AU12R', 'AU01', 'AU02L', 'AU02R']
        }
    ),
    Craig=dict(
        # https://www.tandfonline.com/doi/pdf/10.1080/02699930701516759?needAccess=true
        confused={
            0: ['AU04', 'AU07L', 'AU07R'],
            1: ['AU04', 'AU07L', 'AU07R', 'AU12L', 'AU12R']
        },
        bored=['AU43']
    )
)