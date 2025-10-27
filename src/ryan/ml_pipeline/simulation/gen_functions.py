"""This module contains functions that are used to generate data for main.simulation."""


def gen_global_Q(data_package):
    return data_package.df["global_Q"].iloc[-1] + gen_delta_Q(data_package)


def gen_delta_Q(data_package):
    return (data_package.current * data_package.delta_t) / 1000


def gen_global_time_positive_i(data_package):
    return data_package.df["global_time_positive_i"].iloc[-1] + gen_delta_time_positive_i(data_package)


def gen_delta_time_positive_i(data_package):
    return data_package.delta_t if data_package.current > data_package.cutoff_current else 0


def gen_global_time_negative_i(data_package):
    return data_package.df["global_time_negative_i"].iloc[-1] + gen_delta_time_negative_i(data_package)


def gen_delta_time_negative_i(data_package):
    return data_package.delta_t if data_package.current < data_package.cutoff_current else 0


def gen_delta_q_pos_i(data_package):
    return gen_delta_Q(data_package) if data_package.current > data_package.cutoff_current else 0


def gen_global_q_pos_i(data_package):
    return data_package.df["global_q_pos_i"].iloc[-1] + gen_delta_q_pos_i(data_package)


def gen_delta_q_neg_i(data_package):
    return gen_delta_Q(data_package) if data_package.current < data_package.cutoff_current else 0


def gen_global_q_neg_i(data_package):
    return data_package.df["global_q_neg_i"].iloc[-1] + gen_delta_q_neg_i(data_package)


def gen_global_q_posneg_ratio(data_package):
    if gen_global_q_neg_i(data_package) == 0:
        return 1  # Avoid division by zero error
    else:
        return gen_global_q_pos_i(data_package) / gen_global_q_neg_i(data_package)


def gen_delta_i_neg_avg(data_package):
    return data_package.current if data_package.current < data_package.cutoff_current else 0


def gen_delta_i_pos_avg(data_package):
    return data_package.current if data_package.current > data_package.cutoff_current else 0


def gen_delta_i_posneg_avg_ratio(data_package):
    if gen_delta_i_neg_avg(data_package) == 0:
        return 1  # Avoid division by zero error
    else:
        return gen_delta_i_pos_avg(data_package) / gen_delta_i_neg_avg(data_package)
