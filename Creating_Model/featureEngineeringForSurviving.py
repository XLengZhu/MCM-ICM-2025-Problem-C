import numpy as np
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def calculate_features(athletes_df, noc, current_year):
    """
    计算指定国家在指定年份的特征（只使用之前的数据）
    """
    # 选择该国家到上一届为止的所有历史数据
    historical_data = athletes_df[
        (athletes_df['NOC'] == noc) &
        (athletes_df['Year'] < current_year)
        ]

    # 如果是第一次参赛，返回默认特征
    if len(historical_data) == 0:
        return get_default_features()

    features = {}

    # 获取上一届数据
    last_year = historical_data['Year'].max()
    last_year_data = historical_data[historical_data['Year'] == last_year]

    # 1. 参与度特征
    features['total_appearances'] = historical_data['Year'].nunique()
    features['total_athletes'] = len(historical_data['Name'].unique())
    features['unique_events'] = historical_data['Event'].nunique()
    features['unique_sports'] = historical_data['Sport'].nunique()

    # 平均值计算（使用历史数据）
    features['athletes_per_games'] = features['total_athletes'] / features['total_appearances']
    features['events_per_games'] = features['unique_events'] / features['total_appearances']

    # 2. 上一届特征
    if len(last_year_data) > 0:
        # 计算上一届的项目集中度(HHI)
        event_counts = last_year_data['Event'].value_counts()
        total_athletes = len(last_year_data)
        event_shares = event_counts / total_athletes
        features['last_HHI'] = (event_shares ** 2).sum()

        # 上一届运动员和项目数
        features['last_athletes_count'] = len(last_year_data['Name'].unique())
        features['last_unique_events'] = last_year_data['Event'].nunique()
        features['last_unique_sports'] = last_year_data['Sport'].nunique()

        # 性别比例（上一届）
        features['last_female_ratio'] = (last_year_data['Sex'] == 'F').mean()

        # 老将比例（上一届中参加过之前届次的运动员比例）
        previous_athletes = set(historical_data[historical_data['Year'] < last_year]['Name'])
        last_athletes = set(last_year_data['Name'])
        veteran_count = len(last_athletes.intersection(previous_athletes))
        features['last_veteran_ratio'] = veteran_count / len(last_athletes) if len(last_athletes) > 0 else 0
        features['athlete_per_event'] = (features['last_athletes_count'] /
                                         features['last_unique_events'] if features['last_unique_events'] > 0 else 0)
    else:
        features.update({
            'last_HHI': 0,
            'last_athletes_count': 0,
            'last_unique_events': 0,
            'last_unique_sports': 0,
            'last_female_ratio': 0,
            'last_veteran_ratio': 0,
            'athlete_per_event':0
        })

    # 3. 趋势特征（使用历史数据）
    years = sorted(historical_data['Year'].unique())
    if len(years) >= 2:
        # 计算运动员数量的变化趋势
        athletes_by_year = historical_data.groupby('Year')['Name'].nunique()
        features['athlete_growth_rate'] = athletes_by_year.pct_change().mean()

        # 计算项目数量的变化趋势
        events_by_year = historical_data.groupby('Year')['Event'].nunique()
        features['event_growth_rate'] = events_by_year.pct_change().mean()
    else:
        features.update({
            'athlete_growth_rate': 0,
            'event_growth_rate': 0
        })

    # 5. 运动员多样性特征（使用历史数据）
    athlete_events = historical_data.groupby('Name')['Event'].nunique()
    features['historical_events_per_athlete'] = athlete_events.mean()

    return features


def get_default_features():
    """返回默认特征值"""
    return {
        'total_appearances': 0,#
        'total_athletes': 0,#
        'unique_events': 0,#
        'unique_sports': 0,#
        'athletes_per_games': 0,#
        'events_per_games': 0,#
        'last_HHI': 0,#
        'last_athletes_count': 0,#
        'last_unique_events': 0,#
        'last_unique_sports': 0,#
        'last_female_ratio': 0,#
        'last_veteran_ratio': 0,#
        'athlete_growth_rate': 0,#
        'event_growth_rate': 0,#
        'historical_events_per_athlete': 0,
        'athlete_per_event': 0
    }

# 使用示例：
def create_time_varying_dataset(athletes_df, medal_counts_df):
    time_varying_data = []
    years = sorted(athletes_df['Year'].unique())

    # 对每个国家
    for noc in athletes_df['NOC'].unique():
        # 找到该国首次参赛年份和首次获奖年份
        first_year = athletes_df[athletes_df['NOC'] == noc]['Year'].min()
        first_medal_year = medal_counts_df[
            (medal_counts_df['NOC_Code'] == noc) &
            (medal_counts_df['Total'] > 0)
            ]['Year'].min()

        # 获取该国实际参赛的年份
        participated_years = set(athletes_df[athletes_df['NOC'] == noc]['Year'])

        # 对于首次参赛到首次获奖（或最后一届）之间的每一届
        last_year = first_medal_year if pd.notna(first_medal_year) else max(years)
        for year in range(first_year, last_year + 4, 4):
            # 检查是否实际参赛
            if year not in participated_years:
                continue

            # 计算特征
            features = calculate_features(athletes_df, noc, year)

            # 检查特征是否有效
            if any(pd.isna(value) for value in features.values()):
                continue

            # 添加记录
            time_varying_data.append({
                'NOC': noc,
                'Year': year,
                'duration': (year - first_year) // 4,
                'event': 1 if year == first_medal_year else 0,
                **features
            })

    return pd.DataFrame(time_varying_data)


# def split_data_for_cox(time_varying_df):
#     """
#     划分训练集和测试集
#     训练集：2020年之前的所有数据
#     测试集：2020-2024年间获得首枚奖牌的国家的数据
#     """
#     # 获取训练集：2020年之前的所有数据
#     train_data = time_varying_df[time_varying_df['Year'] < 2000]
#
#     # 获取2020-2024年间获得首枚奖牌的国家
#     first_medal_countries = time_varying_df[
#         (time_varying_df['Year'].isin([2000, 2024])) &
#         (time_varying_df['event'] == 1)
#         ]['NOC'].unique()
#
#     # 获取这些国家的所有历史数据作为测试集
#     test_data = time_varying_df[
#         (time_varying_df['Year'].isin([2000, 2024])) &
#         (time_varying_df['NOC'].isin(first_medal_countries))
#         ]
#
#     print(f"训练集大小: {len(train_data)}")
#     print(f"测试集大小: {len(test_data)}")
#     print(f"\n2020-2024年间获得首枚奖牌的国家: {', '.join(first_medal_countries)}")
#
#     # 检查数据集
#     print("\n训练集年份范围:", train_data['Year'].min(), "-", train_data['Year'].max())
#     print("测试集年份范围:", test_data['Year'].min(), "-", test_data['Year'].max())
#
#     # 检查事件分布
#     print("\n训练集事件分布:")
#     print(train_data['event'].value_counts())
#     print("\n测试集事件分布:")
#     print(test_data['event'].value_counts())
#
#     return train_data, test_data

def build_prediction_dataset(athletes_df, medal_counts_df):
    """
    构建2028年预测数据集

    Parameters:
    -----------
    athletes_df : DataFrame
        运动员数据集
    medal_counts_df : DataFrame
        奖牌数据集

    Returns:
    --------
    prediction_df : DataFrame
        用于预测的数据集
    """
    # 1. 找出2024年参赛但从未获得过奖牌的国家
    # 获取2024年参赛的国家
    countries_2024 = athletes_df[athletes_df['Year'] == 2024]['NOC'].unique()

    # 获取历史上获得过奖牌的国家
    medalist_countries = medal_counts_df[
        medal_counts_df['Total'] > 0
        ]['NOC_Code'].unique()

    # 找出目标国家：2024年参赛但从未获得过奖牌的国家
    target_countries = [
        noc for noc in countries_2024
        if noc not in medalist_countries
    ]

    prediction_data = []
    for noc in target_countries:
        # 找到该国首次参赛年份
        first_year = athletes_df[athletes_df['NOC'] == noc]['Year'].min()

        # 计算duration
        duration = (2028 - first_year) // 4

        # 计算特征
        features = calculate_features(athletes_df, noc, 2028)

        # 添加记录
        prediction_data.append({
            'NOC': noc,
            'duration': duration,
            'event': 0,  # 占位符
            **features
        })

    prediction_df = pd.DataFrame(prediction_data)
    return prediction_df

def split_data_for_cox(df):
    """
    按国家而不是按时间分割数据
    """
    # 获取所有获得过奖牌的国家
    medal_countries = df[df['event'] == 1]['NOC'].unique()

    # 随机分割国家
    np.random.seed(42)
    train_countries = np.random.choice(
        medal_countries,
        size=int(len(medal_countries) * 0.8),
        replace=False
    )
    test_countries = [c for c in medal_countries if c not in train_countries]

    # 分割数据
    train_data = df[df['NOC'].isin(train_countries)]
    test_data = df[df['NOC'].isin(test_countries)]
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    # 检查事件分布
    print("\n训练集事件分布:")
    print(train_data['event'].value_counts())
    print("\n测试集事件分布:")
    print(test_data['event'].value_counts())


    return train_data, test_data


medal_counts = pd.read_csv("../summerOly_medal_counts.csv")
athletes = pd.read_csv("../summerOly_athletes.csv")
medal_counts['NOC'] = medal_counts['NOC'].str.strip()
country_to_noc = {
    'United States': 'USA',
    'Greece': 'GRE',
    'Germany': 'GER',
    'France': 'FRA',
    'Great Britain': 'GBR',
    'Hungary': 'HUN',
    'Austria': 'AUT',
    'Australia': 'AUS',
    'Denmark': 'DEN',
    'Switzerland': 'SUI',
    'Mixed team': 'ZZX',  # 非正式队伍，通常表示联合队
    'Belgium': 'BEL',
    'Italy': 'ITA',
    'Cuba': 'CUB',
    'Canada': 'CAN',
    'Spain': 'ESP',
    'Luxembourg': 'LUX',
    'Norway': 'NOR',
    'Netherlands': 'NED',
    'India': 'IND',
    'Bohemia': 'BOH',  # 现捷克的一部分，历史代表
    'Sweden': 'SWE',
    'Australasia': 'ANZ',  # 过去澳大利亚和新西兰的联合队
    'Russian Empire': 'RU1',  # 非正式缩写
    'Finland': 'FIN',
    'South Africa': 'RSA',
    'Estonia': 'EST',
    'Brazil': 'BRA',
    'Japan': 'JPN',
    'Czechoslovakia': 'TCH',  # 捷克斯洛伐克，历史队伍
    'New Zealand': 'NZL',
    'Yugoslavia': 'YUG',
    'Argentina': 'ARG',
    'Uruguay': 'URU',
    'Poland': 'POL',
    'Haiti': 'HAI',
    'Portugal': 'POR',
    'Romania': 'ROU',
    'Egypt': 'EGY',
    'Ireland': 'IRL',
    'Chile': 'CHI',
    'Philippines': 'PHI',
    'Mexico': 'MEX',
    'Latvia': 'LAT',
    'Turkey': 'TUR',
    'Jamaica': 'JAM',
    'Peru': 'PER',
    'Ceylon': 'CEY',  # 现斯里兰卡
    'Trinidad and Tobago': 'TTO',
    'Panama': 'PAN',
    'South Korea': 'KOR',
    'Iran': 'IRI',
    'Puerto Rico': 'PUR',
    'Soviet Union': 'URS',
    'Lebanon': 'LBN',
    'Bulgaria': 'BUL',
    'Venezuela': 'VEN',
    'United Team of Germany': 'EUA',  # 1956-1964年的东西德联合队
    'Iceland': 'ISL',
    'Pakistan': 'PAK',
    'Bahamas': 'BAH',
    'Ethiopia': 'ETH',
    'Formosa': 'TPE',  # 现为中华台北
    'Ghana': 'GHA',
    'Morocco': 'MAR',
    'Singapore': 'SGP',
    'British West Indies': 'BWI',  # 历史联合队
    'Iraq': 'IRQ',
    'Tunisia': 'TUN',
    'Kenya': 'KEN',
    'Nigeria': 'NGR',
    'East Germany': 'GDR',
    'West Germany': 'FRG',
    'Mongolia': 'MGL',
    'Uganda': 'UGA',
    'Cameroon': 'CMR',
    'Taiwan': 'TPE',  # 中华台北
    'North Korea': 'PRK',
    'Colombia': 'COL',
    'Niger': 'NIG',
    'Bermuda': 'BER',
    'Thailand': 'THA',
    'Zimbabwe': 'ZIM',
    'Tanzania': 'TAN',
    'Guyana': 'GUY',
    'China': 'CHN',
    'Ivory Coast': 'CIV',
    'Syria': 'SYR',
    'Algeria': 'ALG',
    'Chinese Taipei': 'TPE',
    'Dominican Republic': 'DOM',
    'Zambia': 'ZAM',
    'Suriname': 'SUR',
    'Costa Rica': 'CRC',
    'Indonesia': 'INA',
    'Netherlands Antilles': 'AHO',  # 历史队伍
    'Senegal': 'SEN',
    'Virgin Islands': 'ISV',
    'Djibouti': 'DJI',
    'Unified Team': 'EUN',  # 1992年的苏联继承国联合队
    'Lithuania': 'LTU',
    'Namibia': 'NAM',
    'Croatia': 'CRO',
    'Independent Olympic Participants': 'IOP',  # 独立运动员
    'Israel': 'ISR',
    'Slovenia': 'SLO',
    'Malaysia': 'MAS',
    'Qatar': 'QAT',
    'Russia': 'RUS',
    'Ukraine': 'UKR',
    'Czech Republic': 'CZE',
    'Kazakhstan': 'KAZ',
    'Belarus': 'BLR',
    'FR Yugoslavia': 'SCG',  # 塞尔维亚和黑山
    'Slovakia': 'SVK',
    'Armenia': 'ARM',
    'Burundi': 'BDI',
    'Ecuador': 'ECU',
    'Hong Kong': 'HKG',
    'Moldova': 'MDA',
    'Uzbekistan': 'UZB',
    'Azerbaijan': 'AZE',
    'Tonga': 'TGA',
    'Georgia': 'GEO',
    'Mozambique': 'MOZ',
    'Saudi Arabia': 'KSA',
    'Sri Lanka': 'SRI',
    'Vietnam': 'VIE',
    'Barbados': 'BAR',
    'Kuwait': 'KUW',
    'Kyrgyzstan': 'KGZ',
    'Macedonia': 'MKD',
    'United Arab Emirates': 'UAE',
    'Serbia and Montenegro': 'SCG',
    'Paraguay': 'PAR',
    'Eritrea': 'ERI',
    'Serbia': 'SRB',
    'Tajikistan': 'TJK',
    'Samoa': 'SAM',
    'Sudan': 'SUD',
    'Afghanistan': 'AFG',
    'Mauritius': 'MRI',
    'Togo': 'TOG',
    'Bahrain': 'BRN',
    'Grenada': 'GRN',
    'Botswana': 'BOT',
    'Cyprus': 'CYP',
    'Gabon': 'GAB',
    'Guatemala': 'GUA',
    'Montenegro': 'MNE',
    'Independent Olympic Athletes': 'IOA',
    'Fiji': 'FIJ',
    'Jordan': 'JOR',
    'Kosovo': 'KOS',
    'ROC': 'ROC',  # 俄罗斯奥委会
    'San Marino': 'SMR',
    'North Macedonia': 'MKD',
    'Turkmenistan': 'TKM',
    'Burkina Faso': 'BUR',
    'Saint Lucia': 'LCA',
    'Dominica': 'DMA',
    'Albania': 'ALB',
    'Cabo Verde': 'CPV',
    'Refugee Olympic Team': 'ROT'
}
# 将国家全名映射到奥委会缩写代码
medal_counts['NOC_Code'] = medal_counts['NOC'].map(country_to_noc)
df = create_time_varying_dataset(athletes,medal_counts)
# print(df.isna().sum())
df.to_csv("./cox_dataset.csv")
train_data, test_data = split_data_for_cox(df)
train_data.to_csv("./cox_train_data.csv")
test_data.to_csv("./cox_test_data.csv")
print(df.columns)
# print(df.head())
predict_data = build_prediction_dataset(athletes,medal_counts)
print(predict_data.isna().sum())
predict_data.to_csv("./predict_data.csv")