import sys

from utils import *

# Paths and filenames of the datasets
path = os.path.dirname(os.path.abspath(__file__))
db_collusion_brazilian = os.path.join(path, 'DB_Collusion_Brazil_processed.csv')
db_collusion_italian = os.path.join(path, 'DB_Collusion_Italy_processed.csv')
db_collusion_american = os.path.join(path, 'DB_Collusion_America_processed.csv')
db_collusion_switzerland_gr_sg = os.path.join(path, 'DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv')
db_collusion_switzerland_ticino = os.path.join(path, 'DB_Collusion_Switzerland_Ticino_processed.csv')
db_collusion_japan = os.path.join(path, 'DB_Collusion_Japan_processed.csv')
db_collusion_all = os.path.join(path, 'DB_Collusion_All_processed.csv')

if __name__ == '__main__':
    
    # The user selectes the dataset to analyse
    dataset = None
    while dataset == None:
        number_input = input('Insert the number to analyse the dataset [brazilian (1), american (2), italian (3), ' \
                         'switzerland_gr_sg (4), switzerland_ticino (5), japan (6), all datasets (7) or exit (0)]: ')
        if number_input == '0': sys.exit(0)
        elif number_input == '1': dataset = 'brazilian'
        elif number_input == '2': dataset = 'american'
        elif number_input == '3': dataset = 'italian'
        elif number_input == '4': dataset = 'switzerland_gr_sg'
        elif number_input == '5': dataset = 'switzerland_ticino'
        elif number_input == '6': dataset = 'japan'       
        elif number_input == '7': dataset = 'all'

    # 1. Get the dataset processed ready to use with the ML algorithms
    df_collusion, predictors, targets = get_dataset(dataset)
    
    # 2. Print information of the processed datasets
    print_description_processed_dataset(df_collusion)
    
    # 3. Print list with the colluded tenders by bidder
    #calculate_colluded_tenders_by_bidder(df_collusion)

    # 4. Print Scatter Matrix
    df_scatter_matrix = shuffle_tenders(df_collusion)
    # Columns to plot
    columns_to_plot = ['Bid_value'] + screens + ['Collusive_competitor_original'] # Collusive_competitor is deleted at the end
    df_scatter_matrix = df_scatter_matrix[columns_to_plot]
    # Replace labels for colors to print the scatter matrix
    if 'Collusive_competitor_original' in df_scatter_matrix:
        df_scatter_matrix['Collusive_competitor'] = df_scatter_matrix['Collusive_competitor_original']
    colors_legend = ['Green', 'Red']
    labels_legend = ['Competitive bid', 'Collusive bid']
    df_color_labels = df_scatter_matrix[['Collusive_competitor']].replace(0, colors_legend[0])
    df_color_labels = df_color_labels[['Collusive_competitor']].replace(1, colors_legend[1])
    list_color_labels = df_color_labels['Collusive_competitor'].values.tolist()
    df_scatter_matrix.drop(columns=['Collusive_competitor'], inplace=True)
    printScatterMatrix(df_scatter_matrix, list_color_labels, colors_legend, labels_legend, dataset)

    # 5. Boxplots of screen variables
    # Check with len of screens
    max_ylim_screens = [0.4, 1.8, 0.3, 12, 4, 4, 1]
    min_ylim_screens = [0, 0, 0, -12, -4, -4, 0]
    step_y_screens = [0.4/10, 1.8/10, 0.3/10, 24/10, 8/10, 8/10, 1/10]
    df_collusion_copy = df_collusion.copy()
    df_collusion_copy['Collusive_competitor_original'].replace(0, 'Comp.', inplace=True)
    df_collusion_copy['Collusive_competitor_original'].replace(1, 'Coll.', inplace=True)
    for index, screen_variable in enumerate(screens):
        print_boxplot(df_collusion_copy, dataset, column_names=screen_variable, groupby='Collusive_competitor_original', min_ylim=min_ylim_screens[index],
                      max_ylim=max_ylim_screens[index], step_y=step_y_screens[index], xlabel='Bids', percentage=True)

    # 6. Histogram or density plot of Number of Bids by Tender. Each plot for collusive tenders and honest tenders.

    df_hist = df_collusion
    if 'Collusive_competitor_original' in df_hist:
        df_hist['Collusive_competitor'] = df_hist['Collusive_competitor_original']
    competitive_bids = df_hist[df_hist['Collusive_competitor'] == 0]['Number_bids']
    collusive_bids = df_hist[df_hist['Collusive_competitor'] != 0]['Number_bids']
    plotTwoHistograms(dataset, competitive_bids, collusive_bids, label_1='competitive bids', label_2='collusive bids', max_range=125, bins=25, max_xlim=125, density=True)

    # 7. Execute algorithm comparison and print table comparison  
    algorithm_comparison(df_collusion, dataset, predictors, targets, ml_algorithms, train_size, repetitions, n_estimators, precision_recall, load_data, save_data, quality_table)
   