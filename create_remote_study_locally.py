import optuna

study = optuna.create_study(
    direction='maximize',
    study_name='amazing_comeback_real',
    storage="postgresql://aaaa:12345678@postgresql-100628-0.cloudclusters.net:10006/optuna_1",
)
