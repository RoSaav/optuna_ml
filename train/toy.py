import optuna

def objective_bi(trial):
    x = trial.suggest_float('x', -100, 100)
    y = trial.suggest_float('y', -50, 50)
    return ( x + y - 10) ** 2

def objective_uni(trial):
    x = trial.suggest_float('x', -100, 100)
    return ( x - 10) ** 2

def optuna_optimize(
    direction : str='maximize',
    n_trials: int=100,
    kind : str='uni',
    ):

    study = optuna.create_study(direction=direction)
    
    if kind == 'uni':
        study.optimize(objective_uni, n_trials=n_trials)
    else:
        study.optimize(objective_bi, n_trials=n_trials)

    print(study.best_params)

    path_img2 = f'img/toy_plot_edf_{kind}_{direction}.png'
    fig2 = optuna.visualization.plot_edf(study)
    fig2.write_image(path_img2,  format='png')

    path_img3 = f'img/toy_plot_optimization_history_{kind}_{direction}.png'
    fig3 = optuna.visualization.plot_optimization_history(study)
    fig3.write_image(path_img3,  format='png')

    path_img4 = f'img/toy_plot_slice_{kind}_{direction}.png'
    fig4 =optuna.visualization.plot_slice(study)
    fig4.write_image(path_img4,  format='png')

    path_img5 = f'img/toy_plot_param_importances_{kind}_{direction}.png'
    fig5 =optuna.visualization.plot_param_importances(study)
    fig5.write_image(path_img5,  format='png')

    return study.best_params

if __name__ == "__main__":

    for d in ['maximize', 'minimize']:
        for  k in ['uni', 'bi']:
            optuna_optimize(
                direction = d,
                n_trials = 1000,
                kind = k,
                )
