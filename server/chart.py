import pandas as pd

from typing import Any, Dict, List


class Charter:
    @staticmethod
    def chart_data(training_data: List[pd.DataFrame]):
        price_chart_data = pd.concat(training_data, ignore_index=True)
        price_chart_data = price_chart_data.pivot_table(
            index='label',
            columns='symbol',
            values='average'
        )
        price_chart_data.reset_index(inplace=True)
        
        price_chart_data = price_chart_data.replace([float('inf'), -float('inf'), float('nan')], None)
        columns = price_chart_data.columns.tolist()
        if 'label' in columns:
            columns.remove('label')
        price_chart_config = {
            column: {
                "label": column,
                "color": f"hsl(var(--chart-{i+1}))"
            } for i, column in enumerate(columns)
        }
        
        return {
            "price_chart_config": price_chart_config,
            "price_chart_data": price_chart_data.to_dict(orient='records')
        }
