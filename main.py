import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


class DrawingPlots:

    def draw_plots(self, path_json: str):
        df = pd.read_json(path_json)

        confusion_matrix = metrics.confusion_matrix(df.gt_corners, df.rb_corners)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[4, 6, 8, 10])
        cm_display.plot()
        plt.title('Confusion matrix')
        plt.savefig(f'./plots/confusion_matrix.png')
        plt.clf()

        for column in ['mean', 'min', 'max']:
            series = df[['gt_corners', column]].groupby('gt_corners')[column].apply(list)
            plt.boxplot(series, labels=df.gt_corners.unique())
            plt.xlabel('gt_corners')
            plt.ylabel(column)
            plt.title(f'Distribution of {column} from gt_cornerns')
            plt.savefig(f'./plots/gt_corners_{column}.png')
            plt.clf()

        for column in ['mean', 'min', 'max']:
            plt.scatter(x=df[f'floor_{column}'], y=df[f'ceiling_{column}'])
            plt.xlabel(f'floor_{column}')
            plt.ylabel(f'ceiling_mean{column}')
            plt.title(f'Dependence between floor_{column} and ceiling_{column}')
            plt.savefig(f'./plots/dep_floor_ceil_{column}.png')
            plt.clf()

        with os.scandir('./plots/') as files:
            path_all_plots = [file.path for file in files if file.name.endswith('.png')]

        return path_all_plots


if __name__ == '__main__':
    dp = DrawingPlots()
    dp.draw_plots('deviation.json')
