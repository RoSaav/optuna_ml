import optuna

def objective(trial):
    x = trial.suggest_float('x', -20, 20)
    return (x - 10) ** 2

def optuna_optimize(
    direction : str='maximize',
    n_trials: int=100
    ):

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)

    path_img2 = f'img/toy_plot_edf_{direction}.png'
    fig2 = optuna.visualization.plot_edf(study)
    fig2.write_image(path_img2,  format='png')

    path_img3 = f'img/toy_plot_optimization_history_{direction}.png'
    fig3 = optuna.visualization.plot_optimization_history(study)
    fig3.write_image(path_img3,  format='png')

    path_img4 = f'img/toy_plot_slice_{direction}.png'
    fig4 =optuna.visualization.plot_slice(study)
    fig4.write_image(path_img4,  format='png')

    path_img5 = f'img/toy_plot_param_importances_{direction}.png'
    fig5 =optuna.visualization.plot_param_importances(study)
    fig5.write_image(path_img5,  format='png')

    return study.best_params

if __name__ == "__main__":

    for dir in ['maximize', 'minimize']:
        optuna_optimize(
            direction = dir,
            n_trials = 1000)
