#!/usr/bin/env python
# coding: utf-8

import logging
import random

import click
import numpy
import requests

from sklearn.linear_model import LinearRegression, LogisticRegression


APPLICATION_ID = "demo"
MIN_BATTLES = 0
FEATURE_NAMES = [
    # "chassis_rotation_speed",
    # "circular_vision_radius",
    # "engine_power",
    # "gun_damage_max",
    # "gun_damage_min",
    # "gun_max_ammo",
    # "gun_piercing_power_max",
    # "gun_piercing_power_min",
    # "gun_rate",
    # "max_health",
    # "radio_distance",
    # "speed_limit",
    # "turret_armor_board",
    # "turret_armor_fedd",
    # "turret_armor_forehead",
    # "turret_rotation_speed",
    # "vehicle_armor_board",
    # "vehicle_armor_fedd",
    # "vehicle_armor_forehead",
    # "weight",
]
TANK_TYPE_ENCODER = {
    "mediumTank": [1, 0, 0, 0, 0],
    "heavyTank": [0, 1, 0, 0, 0],
    "AT-SPG": [0, 0, 1, 0, 0],
    "SPG": [0, 0, 0, 1, 0],
    "lightTank": [0, 0, 0, 0, 1],
}
TANK_LEVEL_ENCODER = {i: [0.0] * (i - 1) + [1.0] + [0.0] * (10 - i) for i in range(1, 11)}
TANK_NATION_ENCODER = {
    "ussr": [1, 0, 0, 0, 0, 0, 0],
    "france": [0, 1, 0, 0, 0, 0, 0],
    "usa": [0, 0, 1, 0, 0, 0, 0],
    "uk": [0, 0, 0, 1, 0, 0, 0],
    "germany": [0, 0, 0, 0, 1, 0, 0],
    "japan": [0, 0, 0, 0, 0, 1, 0],
    "china": [0, 0, 0, 0, 0, 0, 1],
}


@click.command()
@click.argument("account_id", type=int)
def main(account_id: int):
    """
    Test prediction win rate using tank's characteristics.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session = requests.Session()
    account_tanks = session.get("https://api.worldoftanks.ru/wot/account/tanks/", params={
        "application_id": APPLICATION_ID,
        "account_id": account_id,
    }).json()["data"][str(account_id)]
    logging.info("%s account tanks.", len(account_tanks))
    random.seed(1)
    random.shuffle(account_tanks)
    account_tanks = [
        tank for tank in account_tanks
        if tank["statistics"]["battles"] >= MIN_BATTLES
    ]
    logging.info("Filtered %s account tanks.", len(account_tanks))
    tankinfos = session.get("https://api.worldoftanks.ru/wot/encyclopedia/tankinfo/", params={
        "application_id": APPLICATION_ID,
        "tank_id": ",".join(str(tank["tank_id"]) for tank in account_tanks),
    }).json()["data"]
    logging.info("%s tank infos.", len(tankinfos))
    y = numpy.array([
        tank["statistics"]["wins"] / tank["statistics"]["battles"] >= 0.5
        for tank in account_tanks
    ])
    x = []
    for tank in account_tanks:
        tankinfo = tankinfos[str(tank["tank_id"])]
        feature_values = []
        feature_values.extend(tankinfo[feature_name] for feature_name in FEATURE_NAMES)
        feature_values.extend(TANK_TYPE_ENCODER[tankinfo["type"]])
        feature_values.extend(TANK_LEVEL_ENCODER[tankinfo["level"]])
        # feature_values.append(tankinfo["level"])
        feature_values.append(float(tankinfo["is_premium"]))
        # feature_values.extend(MARK_OF_MASTERY_ENCODER[tank["mark_of_mastery"]])
        feature_values.append(tank["mark_of_mastery"])
        # feature_values.append(tank["statistics"]["battles"])
        # feature_values.append(tankinfo["engine_power"] / tankinfo["weight"])
        # feature_values.extend(TANK_NATION_ENCODER[tankinfo["nation"]])
        x.append(feature_values)
    x = numpy.array(x, dtype=numpy.float)
    train_count = 7 * y.size // 10
    x_train, x_test, y_train, y_test = x[:train_count, :], x[train_count:, :], y[:train_count], y[train_count:]
    logging.info("X: %s. Y: %s.", x.shape, y.size)
    logging.info("Train X: %s. Train Y: %s.", x_train.shape, y_train.size)
    logging.info("Test X: %s. Test Y: %s.", x_test.shape, y_test.size)
    model = LogisticRegression().fit(x_train, y_train)
    for i, (tank, _x, _y) in enumerate(zip(account_tanks, x, y)):
        logging.info(
            "[%s] %s: P: %.2f Y: %.2f",
            "Train" if i < train_count else "Test",
            tankinfos[str(tank["tank_id"])]["localized_name"],
            model.predict(_x),
            _y,
        )
    logging.info("Coef: %s", model.coef_)
    logging.info("Train Score: %f.", model.score(x_train, y_train))
    logging.info("Test Score: %f.", model.score(x_test, y_test))


if __name__ == "__main__":
    main()
