from dataclasses import is_dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, re

plt.rcParams["font.family"] = "Times New Roman"

class ResultComparison:
    def __init__(self):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")          # data directory

    # method: summarize results
    def summarize_results(self):
        # directory, datapath
        feas_dir = os.path.join(self.MLmodel_dir, self.feas_name)
        metric_datapath = os.path.join(feas_dir, "evaluation_metric.csv")
        kpis_datapath = os.path.join(feas_dir, "kpi_results.csv")

        #- demand prediction accuracy -#
        metric = pd.read_csv(metric_datapath)
        MAE = metric['MAE'].copy().tolist()
        RMSE = metric['RMSE'].copy().tolist()
        SUM_DEMAND = metric['SUM_DEMAND'].copy().tolist()

        # weighted mean of MAE (for all part among all dealers)
        MAE_weight = sum([SUM_DEMAND[i]*MAE[i] for i in range(len(SUM_DEMAND))])/sum(SUM_DEMAND)
        
        # std of MAE
        MAE_std = (sum([SUM_DEMAND[i]*((MAE[i]-MAE_weight)**2) for i in range(len(MAE))])
                   /sum(SUM_DEMAND))**(1/2)

        # mean of RMSE (for all part among all dealers)
        RMSE_mean = sum(RMSE)/len(RMSE)

        # std of RMSE
        RMSE_sigma = (sum([(RMSEi-RMSE_mean)**2 for RMSEi in RMSE])/len(RMSE))**(1/2)

        #- kpis -#
        kpis = pd.read_csv(kpis_datapath)
        total_costs = kpis['total_costs']
        ISL = kpis['ISL']
        total_stockouts = kpis['total_stockouts']
        total_demand = kpis['total_demand']
        immediate_fulfilled = kpis['immediate_fulfilled']
        backorder_fulfileed = kpis['backorder_fulfileed']

        # Total cost
        TotalCost_all = sum(total_costs)
        # mean of Total Cost
        TotalCost_mean = sum(total_costs)/len(total_costs)
        # std of Total Cost
        TotalCost_std = (sum([(total_costi-TotalCost_mean)**2 for total_costi in total_costs])
                         /len(total_costs))**(1/2)

        # ISL for all parts
        ISL_all = sum(immediate_fulfilled)/sum(total_demand)
        # mean of ISL
        ISL_mean = sum([immediate_fulfilled[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of ISL
        ISL_std = (sum([(immediate_fulfilled[i]/total_demand[i]-ISL_mean)**2 for i in range(len(total_demand))])
                   /len(total_demand))**(1/2)

        # Stockouts rate for all parts
        StockoutsRate_all = sum(total_stockouts)/sum(total_demand)
        # mean of stockout rate
        StockoutsRate_mean = sum([total_stockouts[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of stockouts rate
        StockoutsRate_std = (sum([(total_stockouts[i]/total_demand[i]-StockoutsRate_mean)**2 for i in range(len(total_demand))])
                             /len(total_demand))**(1/2)

        # Cost per demand
        CostPerDemand = sum(total_costs)/sum(total_demand)
        # mean of Cost per demand
        CostPerDemand_mean = sum([total_costs[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of Cost per demand
        CostPerDemand_std = (sum([(total_costs[i]/total_demand[i]-CostPerDemand_mean)**2 for i in range(len(total_demand))])
                              /len(total_demand))**(1/2)

        # record
        summary_df = pd.DataFrame(columns=["MAE_weight",
                                           "sigma(MAE)",
                                           "RMSE_mean",
                                           "sigma(RMSE)",
                                           "TotalCost_all",
                                           "TotalCost_mean",
                                           "sigma(TotalCost)",
                                           "ISL_all",
                                           "ISL_mean",
                                           "sigma(ISL)",
                                           "StockoutsRate_all",
                                           "StockoutsRate_mean",
                                           "sigma(StockoutsRate)",
                                           "CostPerDemand",
                                           "CostPerDemand_mean",
                                           "sigma(CostPerDemand)"])
        summary_df.loc[self.MLmodel] = [MAE_weight,
                                        MAE_std,
                                        RMSE_mean,
                                        RMSE_sigma,
                                        TotalCost_all,
                                        TotalCost_mean,
                                        TotalCost_std,
                                        ISL_all,
                                        ISL_mean,
                                        ISL_std,
                                        StockoutsRate_all,
                                        StockoutsRate_mean,
                                        StockoutsRate_std,
                                        CostPerDemand,
                                        CostPerDemand_mean,
                                        CostPerDemand_std]
        summary_datapath = os.path.join(feas_dir, "summary_results.csv")
        summary_df.to_csv(summary_datapath)


    # method: compare multiple models
    def visual_single_feature_results(self, feas_name, model_list):
        # load results data
        all_results = pd.DataFrame()
        for model in model_list:
            # directory path, file path
            model_dir = os.path.join(self.data_dir, model)
            feas_dir = os.path.join(model_dir, feas_name)
            datapath = os.path.join(feas_dir, "summary_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            all_results = pd.concat([all_results, summary_results.loc[[model]].copy()])
        
        print(all_results.columns)

        #- Demand Prediction Results -#
        # MAE
        idx = all_results.index
        y = all_results['MAE_weight']
        err = all_results['sigma(MAE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('MAE (weighted mean)')
        plt.title('Comparison of MAE by Model')
        plt.show()

        # RSME
        idx = all_results.index
        y = all_results['RMSE_mean']
        err = all_results['sigma(RMSE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('RSME (mean)')
        plt.title('Comparison of RSME by Model')
        plt.show()

        # R2
        idx = all_results.index
        y = all_results['R2_mean']
        err = all_results['sigma(R2)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('R2 (mean)')
        plt.title('Comparison of R2 by Model')
        plt.show()

        # IAE
        idx = all_results.index
        y = all_results['IAE_mean']
        err = all_results['sigma(IAE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('IAE (mean)')
        plt.title('Comparison of IAE by Model')
        plt.show()


        #- KPIs -#
        # TotalCost
        idx = all_results.index
        y = all_results['TotalCost_all']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Total Cost')
        plt.title('Comparison of Total Cost by Model')
        plt.show()

        # mean of total cost
        idx = all_results.index
        y = all_results['TotalCost_mean']
        err = all_results['sigma(TotalCost)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Total Cost')
        plt.title('Comparison of Mean Total Cost by Model')
        plt.show()

        # ISL
        idx = all_results.index
        y = all_results['ISL_mean']
        err = all_results['sigma(ISL)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Immediate Service Level')
        plt.title('Comparison of ISL by Model')
        plt.show()
            
        # Stockouts Rate
        idx = all_results.index
        y = all_results['StockoutsRate_mean']
        err = all_results['sigma(StockoutsRate)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Stockouts Rate')
        plt.title('Comparison of Stockouts Rate by Model')
        plt.show()

        # Cost Per Demand
        idx = all_results.index
        y = all_results['CostPerDemand_mean']
        err = all_results['sigma(CostPerDemand)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Cost Per Demand')
        plt.title('Comparison of Cost Per Demand by Model')
        plt.show()


    # method: compare multiple models with multiple types of feature
    def visual_multiple_feature_results(self, feas_list, model_list, TSA_list, Noise_list):
        # load results data
        all_results = pd.DataFrame()
        for feas_name in feas_list:
            for model in model_list:
                # directory path, file path
                model_dir = os.path.join(self.data_dir, model)
                feas_dir = os.path.join(model_dir, feas_name)
                datapath = os.path.join(feas_dir, "summary_results.csv")

                # summary results
                summary_results = pd.read_csv(datapath, index_col=0)
                row = summary_results.loc[[model]].copy()
                # row['feature'] = feas_name
                row.index = [str(model)+"\n"+str(feas_name)]
                all_results = pd.concat([all_results, row.copy()])
        
        # TSA
        for TSA in TSA_list:
            # directory path, file path
            model_dir = os.path.join(self.data_dir, TSA)
            datapath = os.path.join(model_dir, "summary_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            row = summary_results.loc[[TSA]].copy()
            all_results = pd.concat([all_results, row.copy()])
        
        # Noise
        for noise in Noise_list:
            # directory path, file path
            noise_name = "Noise_"+str(noise)
            model_dir = os.path.join(self.data_dir, noise_name)
            datapath = os.path.join(model_dir, "summary_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            row = summary_results.loc[[noise_name]].copy()
            all_results = pd.concat([all_results, row.copy()])

        #- Demand Prediction Results -#
        # MAE
        idx = all_results.index
        y = all_results['MAE_weight']
        err = all_results['sigma(MAE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('MAE')
        plt.title('Comparison of MAE by Model')
        plt.show()

        # RSME
        idx = all_results.index
        y = all_results['RMSE_mean']
        err = all_results['sigma(RMSE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('RMSE')
        plt.title('Comparison of RMSE by Model')
        plt.show()

        # R2
        idx = all_results.index
        y = all_results['R2_mean']
        err = all_results['sigma(R2)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('R2')
        plt.title('Comparison of R2 by Model')
        plt.show()

        # IAE
        idx = all_results.index
        y = all_results['IAE_mean']
        err = all_results['sigma(IAE)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('IAE')
        plt.title('Comparison of IAE by Model')
        plt.show()


        #- KPIs -#
        # TotalCost
        idx = all_results.index
        y = all_results['TotalCost_all']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Total Cost')
        plt.title('Comparison of Total Cost by Model')
        plt.show()

        # mean of total cost
        idx = all_results.index
        y = all_results['TotalCost_mean']
        err = all_results['sigma(TotalCost)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Total Cost')
        plt.title('Comparison of Mean Total Cost by Model')
        plt.show()

        # ISL
        idx = all_results.index
        y = all_results['ISL_mean']
        err = all_results['sigma(ISL)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Immediate Service Level')
        plt.title('Comparison of ISL by Model')
        plt.show()
            
        # Stockouts Rate
        idx = all_results.index
        y = all_results['StockoutsRate_mean']
        err = all_results['sigma(StockoutsRate)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Stockouts Rate')
        plt.title('Comparison of Stockouts Rate by Model')
        plt.show()

        # Cost Per Demand
        idx = all_results.index
        y = all_results['CostPerDemand_mean']
        err = all_results['sigma(CostPerDemand)']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='orange')
        plt.errorbar(idx, y, yerr=err, fmt='none', ecolor='black', capsize=5)
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Mean of Cost Per Demand')
        plt.title('Comparison of Cost Per Demand by Model')
        plt.show()
    

    # method: compare multiple models with multiple types of feature
    def visual_cost_details(self, Noise_list):
        # load results data
        all_results = pd.DataFrame()
        # Noise
        for noise in Noise_list:
            # directory path, file path
            noise_name = "Noise_"+str(noise)
            model_dir = os.path.join(self.data_dir, noise_name)
            datapath = os.path.join(model_dir, "summary_detailed_cost_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            row = summary_results.loc[[noise_name]].copy()
            all_results = pd.concat([all_results, row.copy()])

        print(all_results)

        #- Detailed Costs Results -#
        # Holding Cost
        idx = all_results.index
        y = all_results['hold_sum']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Holding Cost')
        # plt.title('Comparison of MAE by Model')
        plt.show()

        # Order Cost
        idx = all_results.index
        y = all_results['order_sum']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Order Cost')
        # plt.title('Comparison of MAE by Model')
        plt.show()

        # Transport Cost
        idx = all_results.index
        y = all_results['transport_sum']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Transport Cost')
        # plt.title('Comparison of MAE by Model')
        plt.show()

        # Return Cost
        idx = all_results.index
        y = all_results['return_sum']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Return Cost')
        # plt.title('Comparison of MAE by Model')
        plt.show()

        # Badwill Cost
        idx = all_results.index
        y = all_results['badwill_sum']

        plt.figure(figsize=(10,6))
        bars = plt.bar(idx, y, color='skyblue')
        for bar, val in zip(bars, y):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{val:.3f}', 
                    ha='center', va='bottom')
        plt.ylabel('Badwill Cost')
        # plt.title('Comparison of MAE by Model')
        plt.show()


    # method: compare multiple models with multiple types of feature
    def visual_corr_coef(self, feas_list, model_list, TSA_list, Noise_list):
        # load results data
        all_results = pd.DataFrame()
        for feas_name in feas_list:
            for model in model_list:
                # directory path, file path
                model_dir = os.path.join(self.data_dir, model)
                feas_dir = os.path.join(model_dir, feas_name)
                datapath = os.path.join(feas_dir, "CorrCoef.csv")

                # correlation coefficient results
                corr_matrix = pd.read_csv(datapath, index_col=0)
                
                # visualization
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,          # 各セルに数値を表示
                    fmt=".2f",           # 小数点2桁
                    cmap="coolwarm",     # カラーマップ（例：coolwarm）
                    vmin=-1, vmax=1,     # 相関係数の範囲：-1〜1
                    center=0             # カラーマップの中心を 0 に設定
                )
                plt.title("Correlation Matrix of Metrics & KPIs ("+str(model)+"_"+str(feas_name)+")")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
        
        # TSA
        for TSA in TSA_list:
            # directory path, file path
            model_dir = os.path.join(self.data_dir, TSA)
            datapath = os.path.join(model_dir, "CorrCoef.csv")

            # correlation coefficient results
            corr_matrix = pd.read_csv(datapath, index_col=0)
            
            # visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,          # 各セルに数値を表示
                fmt=".2f",           # 小数点2桁
                cmap="coolwarm",     # カラーマップ（例：coolwarm）
                vmin=-1, vmax=1,     # 相関係数の範囲：-1〜1
                center=0             # カラーマップの中心を 0 に設定
            )
            plt.title("Correlation Matrix of Metrics & KPIs ("+str(TSA)+")")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
        
        # Noise
        for noise in Noise_list:
            # directory path, file path
            noise_name = "Noise_"+str(noise)
            model_dir = os.path.join(self.data_dir, noise_name)
            datapath = os.path.join(model_dir, "CorrCoef.csv")

            # correlation coefficient results
            corr_matrix = pd.read_csv(datapath, index_col=0)
            
            # visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,          # 各セルに数値を表示
                fmt=".2f",           # 小数点2桁
                cmap="coolwarm",     # カラーマップ（例：coolwarm）
                vmin=-1, vmax=1,     # 相関係数の範囲：-1〜1
                center=0             # カラーマップの中心を 0 に設定
            )
            plt.title("Correlation Matrix of Metrics & KPIs ("+str(noise_name)+")")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

    # method: plot scatter 
    def plot_scatter_trade_off(self, feas_list, model_list, TSA_list, Noise_list, Item_list):
        from sklearn.preprocessing import MinMaxScaler
        import scipy.stats as stats
        scaler = MinMaxScaler()

        # ML model
        for feas_name in feas_list:
            for model in model_list:
                print("model:"+str(model))
                # directory, datapath
                model_dir = os.path.join(self.data_dir, model)
                feas_dir = os.path.join(model_dir, feas_name)

                # prediction accuracy
                metric_datapath = os.path.join(feas_dir, "evaluation_metric.csv")
                metric = pd.read_csv(metric_datapath)

                # KPIs
                kpis_datapath = os.path.join(feas_dir, "kpi_results.csv")
                kpis = pd.read_csv(kpis_datapath)

                # data frame for plotting
                df_plot = pd.DataFrame({
                    "MAE": scaler.fit_transform(metric['MAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "RMSE": scaler.fit_transform(metric['RMSE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "R2": scaler.fit_transform(metric['R2'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "IAE": scaler.fit_transform(metric['IAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "total_costs": scaler.fit_transform(kpis['total_costs'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "ISL": scaler.fit_transform(kpis['ISL'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    "total_stockouts": scaler.fit_transform(kpis['total_stockouts'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    'total_demand': scaler.fit_transform(kpis['total_demand'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    'immediate_fulfilled': scaler.fit_transform(kpis['immediate_fulfilled'].values.reshape(-1, 1)).reshape(1, -1)[0],
                    'backorder_fulfileed': scaler.fit_transform(kpis['backorder_fulfileed'].values.reshape(-1, 1)).reshape(1, -1)[0]
                })

                for item in Item_list:
                    item0 = item[0]
                    item1 = item[1]

                    plt.figure(figsize=(8,6))
                    sns.regplot(
                        data = df_plot,
                        x    = item0,
                        y    = item1,
                        scatter_kws={'alpha':0.7, 'edgecolor':'k'},
                        line_kws   ={'color':'red'}
                    )

                    # 回帰係数を計算
                    x = df_plot[item0].values
                    y = df_plot[item1].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    # 回帰式を図に表示
                    eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f})'
                    plt.text(
                        x.max()*0.6,    # 表示位置（x座標）
                        y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
                        eq_text,
                        fontsize=12,
                        color='blue'
                    )

                    plt.xlabel(str(item0))
                    plt.ylabel(str(item1))
                    # plt.title(str(model)+': Scatter Plot with Regression Line: '+str(item0)+' vs '+str(item1))
                    plt.show()
        

        # TSA
        for TSA in TSA_list:
            print("TSA:"+str(TSA))
            # directory, datapath
            model_dir = os.path.join(self.data_dir, TSA)

            # prediction accuracy
            metric_datapath = os.path.join(model_dir, "evaluation_metric.csv")
            metric = pd.read_csv(metric_datapath)

            # KPIs
            kpis_datapath = os.path.join(model_dir, "kpi_results.csv")
            kpis = pd.read_csv(kpis_datapath)

            # data frame for plotting
            df_plot = pd.DataFrame({
                "MAE": scaler.fit_transform(metric['MAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "RMSE": scaler.fit_transform(metric['RMSE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "R2": scaler.fit_transform(metric['R2'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "IAE": scaler.fit_transform(metric['IAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "total_costs": scaler.fit_transform(kpis['total_costs'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "ISL": scaler.fit_transform(kpis['ISL'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "total_stockouts": scaler.fit_transform(kpis['total_stockouts'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'total_demand': scaler.fit_transform(kpis['total_demand'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'immediate_fulfilled': scaler.fit_transform(kpis['immediate_fulfilled'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'backorder_fulfileed': scaler.fit_transform(kpis['backorder_fulfileed'].values.reshape(-1, 1)).reshape(1, -1)[0]
            })

            for item in Item_list:
                item0 = item[0]
                item1 = item[1]

                plt.figure(figsize=(8,6))
                sns.regplot(
                    data = df_plot,
                    x    = item0,
                    y    = item1,
                    ci   = None,        # 信頼区間を表示しない（省略可）
                    scatter_kws={'alpha':0.7, 'edgecolor':'k'},
                    line_kws   ={'color':'red'}
                )

                # 回帰係数を計算
                x = df_plot[item0].values
                y = df_plot[item1].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # 回帰式を図に表示
                eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f})'
                plt.text(
                    x.max()*0.6,    # 表示位置（x座標）
                    y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
                    eq_text,
                    fontsize=12,
                    color='blue'
                )

                plt.xlabel(str(item0))
                plt.ylabel(str(item1))
                plt.title(str(TSA)+': Scatter Plot with Regression Line: '+str(item0)+' vs '+str(item1))
                plt.show()


        # Noise
        for noise in Noise_list:
            print("noise:"+str(noise))
            # directory, datapath
            noise_name = "Noise_"+str(noise)
            model_dir = os.path.join(self.data_dir, noise_name)

            # prediction accuracy
            metric_datapath = os.path.join(model_dir, "evaluation_metric.csv")
            metric = pd.read_csv(metric_datapath)

            # KPIs
            kpis_datapath = os.path.join(model_dir, "kpi_results.csv")
            kpis = pd.read_csv(kpis_datapath)

            # data frame for plotting
            df_plot = pd.DataFrame({
                "MAE": scaler.fit_transform(metric['MAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "RMSE": scaler.fit_transform(metric['RMSE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "R2": scaler.fit_transform(metric['R2'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "IAE": scaler.fit_transform(metric['IAE'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "total_costs": scaler.fit_transform(kpis['total_costs'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "ISL": scaler.fit_transform(kpis['ISL'].values.reshape(-1, 1)).reshape(1, -1)[0],
                "total_stockouts": scaler.fit_transform(kpis['total_stockouts'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'total_demand': scaler.fit_transform(kpis['total_demand'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'immediate_fulfilled': scaler.fit_transform(kpis['immediate_fulfilled'].values.reshape(-1, 1)).reshape(1, -1)[0],
                'backorder_fulfileed': scaler.fit_transform(kpis['backorder_fulfileed'].values.reshape(-1, 1)).reshape(1, -1)[0]
            })

            for item in Item_list:
                item0 = item[0]
                item1 = item[1]

                plt.figure(figsize=(8,6))
                sns.regplot(
                    data = df_plot,
                    x    = item0,
                    y    = item1,
                    ci   = None,        # 信頼区間を表示しない（省略可）
                    scatter_kws={'alpha':0.7, 'edgecolor':'k'},
                    line_kws   ={'color':'red'}
                )

                # 回帰係数を計算
                x = df_plot[item0].values
                y = df_plot[item1].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # 回帰式を図に表示
                eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f})'
                plt.text(
                    x.max()*0.6,    # 表示位置（x座標）
                    y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
                    eq_text,
                    fontsize=12,
                    color='blue'
                )

                plt.xlabel(str(item0))
                plt.ylabel(str(item1))
                plt.title(str(noise_name)+': Scatter Plot with Regression Line: '+str(item0)+' vs '+str(item1))
                plt.show()
    
    
    
    # method: compare multiple models with multiple types of feature
    def scatter_multiple_feature_results(self, feas_list, model_list, TSA_list, Noise_list):
        from sklearn.preprocessing import MinMaxScaler
        import scipy.stats as stats
        scaler = MinMaxScaler()
        
        # load results data
        all_results = pd.DataFrame()
        for feas_name in feas_list:
            for model in model_list:
                # directory path, file path
                model_dir = os.path.join(self.data_dir, model)
                feas_dir = os.path.join(model_dir, feas_name)
                datapath = os.path.join(feas_dir, "summary_results.csv")

                # summary results
                summary_results = pd.read_csv(datapath, index_col=0)
                row = summary_results.loc[[model]].copy()
                # row['feature'] = feas_name
                row.index = [str(model)+"\n"+str(feas_name)]
                all_results = pd.concat([all_results, row.copy()])
        
        # TSA
        for TSA in TSA_list:
            # directory path, file path
            model_dir = os.path.join(self.data_dir, TSA)
            datapath = os.path.join(model_dir, "summary_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            row = summary_results.loc[[TSA]].copy()
            all_results = pd.concat([all_results, row.copy()])
        
        # Noise
        for noise in Noise_list:
            # directory path, file path
            noise_name = "Noise_"+str(noise)
            model_dir = os.path.join(self.data_dir, noise_name)
            datapath = os.path.join(model_dir, "summary_results.csv")

            # summary results
            summary_results = pd.read_csv(datapath, index_col=0)
            row = summary_results.loc[[noise_name]].copy()
            all_results = pd.concat([all_results, row.copy()])

        print(all_results.columns)

        print(all_results.index)
        all_results = pd.DataFrame({
            "LABEL": all_results.index.to_numpy(),
            "MAE": scaler.fit_transform(all_results['MAE_weight'].values.reshape(-1, 1)).reshape(1, -1)[0],
            "RMSE": scaler.fit_transform(all_results['RMSE_mean'].values.reshape(-1, 1)).reshape(1, -1)[0],
            "R2": scaler.fit_transform(all_results['R2_mean'].values.reshape(-1, 1)).reshape(1, -1)[0],
            "IAE": scaler.fit_transform(all_results['IAE_mean'].values.reshape(-1, 1)).reshape(1, -1)[0],
            "TotalCost": scaler.fit_transform(all_results['TotalCost_all'].values.reshape(-1, 1)).reshape(1, -1)[0],
        })

        #- Demand Prediction Results -#
        # MAE VS COST
        idx = all_results
        item0 = 'MAE'
        item1 = 'TotalCost'

        plt.figure(figsize=(8,6))
        ax = sns.regplot(
            data = all_results,
            x    = item0,
            y    = item1,
            ci   = None,        # 信頼区間を表示しない（省略可）
            scatter_kws={'alpha':0.7, 'edgecolor':'k'},
            line_kws   ={'color':'red'}
        )

        for idx, row in all_results.iterrows():
            ax.text(
                row[item0],        # x の位置
                row[item1],        # y の位置
                str(row["LABEL"]),  # ラベル文字列
                fontsize=9,
                ha='right',        # 水平位置揃え
                va='bottom'        # 垂直位置揃え
            )

        # 回帰係数を計算
        x = all_results[item0].values
        y = all_results[item1].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 回帰式を図に表示
        eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f}, r={r_value:.2f})'
        plt.text(
            x.max()*0.6,    # 表示位置（x座標）
            y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
            eq_text,
            fontsize=12,
            color='blue'
        )

        plt.xlabel(str(item0))
        plt.ylabel(str(item1))
        # plt.title("Scatter Plot with Regression Line: "+str(item0)+' vs '+str(item1))
        plt.show()

        # RMSE VS COST
        idx = all_results
        item0 = 'RMSE'
        item1 = 'TotalCost'

        plt.figure(figsize=(8,6))
        ax = sns.regplot(
            data = all_results,
            x    = item0,
            y    = item1,
            ci   = None,        # 信頼区間を表示しない（省略可）
            scatter_kws={'alpha':0.7, 'edgecolor':'k'},
            line_kws   ={'color':'red'}
        )
        for idx, row in all_results.iterrows():
            ax.text(
                row[item0],        # x の位置
                row[item1],        # y の位置
                str(row["LABEL"]),  # ラベル文字列
                fontsize=9,
                ha='right',        # 水平位置揃え
                va='bottom'        # 垂直位置揃え
            )

        # 回帰係数を計算
        x = all_results[item0].values
        y = all_results[item1].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 回帰式を図に表示
        eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f}, r={r_value:.2f})'
        plt.text(
            x.max()*0.6,    # 表示位置（x座標）
            y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
            eq_text,
            fontsize=12,
            color='blue'
        )

        plt.xlabel(str(item0))
        plt.ylabel(str(item1))
        # plt.title("Scatter Plot with Regression Line: "+str(item0)+' vs '+str(item1))
        plt.show()


        # R2 VS COST
        idx = all_results
        item0 = 'R2'
        item1 = 'TotalCost'

        plt.figure(figsize=(8,6))
        ax = sns.regplot(
            data = all_results,
            x    = item0,
            y    = item1,
            ci   = None,        # 信頼区間を表示しない（省略可）
            scatter_kws={'alpha':0.7, 'edgecolor':'k'},
            line_kws   ={'color':'red'}
        )

        for idx, row in all_results.iterrows():
            ax.text(
                row[item0],        # x の位置
                row[item1],        # y の位置
                str(row["LABEL"]),  # ラベル文字列
                fontsize=9,
                ha='right',        # 水平位置揃え
                va='bottom'        # 垂直位置揃え
            )

        # 回帰係数を計算
        x = all_results[item0].values
        y = all_results[item1].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 回帰式を図に表示
        eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f}, r={r_value:.2f})'
        plt.text(
            x.max()*0.6,    # 表示位置（x座標）
            y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
            eq_text,
            fontsize=12,
            color='blue'
        )

        plt.xlabel(str(item0))
        plt.ylabel(str(item1))
        # plt.title("Scatter Plot with Regression Line: "+str(item0)+' vs '+str(item1))
        plt.show()

        # IAE VS COST
        idx = all_results
        item0 = 'IAE'
        item1 = 'TotalCost'

        plt.figure(figsize=(8,6))
        ax = sns.regplot(
            data = all_results,
            x    = item0,
            y    = item1,
            ci   = None,        # 信頼区間を表示しない（省略可）
            scatter_kws={'alpha':0.7, 'edgecolor':'k'},
            line_kws   ={'color':'red'}
        )

        for idx, row in all_results.iterrows():
            ax.text(
                row[item0],        # x の位置
                row[item1],        # y の位置
                str(row["LABEL"]),  # ラベル文字列
                fontsize=9,
                ha='right',        # 水平位置揃え
                va='bottom'        # 垂直位置揃え
            )

        # 回帰係数を計算
        x = all_results[item0].values
        y = all_results[item1].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 回帰式を図に表示
        eq_text = f'y = {intercept:.3f} + {slope:.3f} x  (R² = {r_value**2:.2f}, r={r_value:.2f})'
        plt.text(
            x.max()*0.6,    # 表示位置（x座標）
            y.min() + (y.max()-y.min())*0.1,  # 表示位置（y座標）
            eq_text,
            fontsize=12,
            color='blue'
        )

        plt.xlabel(str(item0))
        plt.ylabel(str(item1))
        # plt.title("Scatter Plot with Regression Line: "+str(item0)+' vs '+str(item1))
        plt.show()

    