# StatArb Agent: OPTICS Pairs Trading + RL Sizing

Исследовательский торговый агент для statistical arbitrage. Сначала строится честная rule-based стратегия парного трейдинга, затем поверх нее можно обучать RL-модель для изменения размера позиции. RL не выбирает пары и не должен спасать плохой baseline.

## Суть Стратегии

Логика взята из исследований по clustering-based pairs trading:

- rolling-схема: 36 месяцев formation window -> 1 месяц trading;
- кластеризация акций через OPTICS;
- выбор пар только из одного кластера;
- расчет hedge ratio, z-score и направления сделки только на прошлом formation window;
- торговля следующего месяца без доступа к будущим данным;
- RL используется только как sizing-layer после rule-based сигналов.

Текущий baseline:

```text
distance_method: pca_distance
signal_direction: adaptive
max_portfolio_pairs: 25
min_formation_score: 1.7
entry_z / exit_z: 2.0 / 0.5
```

`min_formation_score` означает: пара допускается к торговле в следующем месяце только если на предыдущем 36-месячном окне ее rule-based сигнал был достаточно сильным.

## Текущий Результат

Последняя честная проверка через Docker:

```text
Rule-based Sharpe: 1.48
Final NAV: 1.3758
Total Return: 37.58%
Annual Return: 14.62%
Max Drawdown: 10.41%
Profit Factor: 1.58
Trading Days: 569
Coverage: 27 торговых месяцев, в среднем 1.5 пары
```

Train/test:

```text
Train Sharpe: 1.83
Test Sharpe: 0.37
Full Sharpe: 1.48
```


## Архитектура

```text
config/base.yaml          основные параметры
run_rl.py                 главный запуск в Docker
src/data/                 загрузка и кеширование данных Stooq
src/clustering/           OPTICS и distance matrices
src/pairs/                выбор пар внутри кластеров
src/backtest/             rolling backtest и signal dataset
src/rl/                   Gym env и обучение Stable-Baselines3
src/pipeline/run.py       orchestration end-to-end
scripts/evaluate_backtest.py  отчет по метрикам
scripts/diagnose.py           sanity-check данных
```

Данные и результаты лежат в `data/`.

## Как Запускать

Запустить контейнеры:

```bash
docker compose up -d --build
```

Проверить rule-based стратегию без RL:

```bash
docker compose exec -e VERIFY_ONLY=1 app python run_rl.py
```

Посмотреть результат:

```bash
docker compose exec app python scripts/evaluate_backtest.py
docker compose exec app python scripts/diagnose.py
```

Проверить синтаксис:

```bash
docker compose exec app python -m compileall -q src scripts
```

## Как Переобучить Модель

Полный запуск с RL:

```bash
docker compose exec app python run_rl.py
```

Что произойдет:

1. Пайплайн заново построит features, clusters, pairs и rule-based backtest.
2. Если rule-based Sharpe отрицательный, RL не обучается.
3. Если baseline положительный, обучается RL на rolling pair episodes.
4. `data/rl_test_nav.csv` сохраняется для диагностики.
5. `data/rl_model.zip` сохраняется только если RL Sharpe на OOS лучше rule-based Sharpe на том же периоде.

Если `data/rl_model.zip` отсутствует после запуска, это нормально: значит RL не улучшил baseline и модель была отброшена.

## Сильные Стороны

- Нет торговли вне своего out-of-sample месяца.
- Formation/trading схема близка к исследованиям.
- Исправлен look-ahead в pair selection.
- Хороший full Sharpe достигается rule-based слоем, без необходимости RL.
- Есть защитный gate: слабый baseline не передается в RL как будто это рабочая стратегия.

## Слабые Стороны

- Высокий Sharpe достигнут строгим фильтром `min_formation_score`, поэтому coverage низкий.
- Стратегия торгует мало месяцев и мало пар, диверсификация ограничена.
- Test Sharpe заметно ниже train Sharpe, значит есть риск переобучения фильтра.
- RL пока не доказал улучшение baseline.
- Результаты чувствительны к universe, источнику данных, комиссиям и режиму рынка.

## Рыночный Режим

Стратегия лучше подходит для режимов, где:

- связи внутри секторов и кластеров устойчивы;
- spread между похожими бумагами возвращается к норме;
- нет длительного однонаправленного разрыва между акциями пары;
- волатильность достаточная для входов, но без частых структурных сломов.

Стратегия хуже работает, когда:

- рынок резко меняет лидеров;
- отдельные акции уходят в длительный тренд относительно кластера;
- корреляции ломаются;
- formation window хорошо выглядел в прошлом, но связь пары исчезла в следующем месяце.

## Основные Артефакты

- `data/pairs.csv`: выбранные пары по месяцам.
- `data/rule_backtest.csv`: rule-based NAV, PnL, turnover, число пар.
- `data/rl_test_nav.csv`: OOS NAV RL, если запускался full pipeline.
- `data/rl_model.zip`: сохраненная RL-модель, только если она обошла rule-based OOS.

