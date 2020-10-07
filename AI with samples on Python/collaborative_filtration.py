import argparse
import json
import numpy as np

from compute_scores import pearson_score


def build_arg_parser():
    """Функция для парсинга входных аргументов, а данном случае
    это имя пользователя."""
    parser = argparse.ArgumentParser(description=
                                     'Найти пользователя, который похож на входного пользователя')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


def find_similar_users(dataset, user, num_users):
    """Функция, которая будет находить в наборе данных пользщователей,
    аналогичных указанному. Если информация об указанном пользователе
    отсутствует, будет генерироваться исклчение."""
    if user not in dataset:
        raise TypeError('Не могу найти пользователя ' + user + 'в датасет')
# Вычисление сходства по Пирсону между
# указанным пользователем и остальными
# пользователями в наборе данных.

    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])

# Сортировка оценок по убыванию
    scores_sorted = np.argsort(scores[:, 1])[::-1]

# извлечение оценок первых  'num_users'
    top_users = scores_sorted[:num_users]
    return scores[top_users]


if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

# Загружаем данные из файла рейтинга
ratings_file = 'ratings.json'

with open(ratings_file, 'r') as f:
    data = json.loads(f.read())

# Находим первых N пользователей, аналогичных указанному с помощью аргумента.
N = 3

print('Пользователи похожие на ' + user + ':\n')
similar_users = find_similar_users(data, user, N)
print('Индекс\t\t\tпохожести пользователя')
print('-' * 41)
for item in similar_users:
    print(item[0], '\t\t', round(float(item[1]), 2))

