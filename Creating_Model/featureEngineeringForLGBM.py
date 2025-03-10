import pandas as pd
import numpy as np
from typing import Dict, List


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

def calculate_features_for_lgb(
        medal_counts_df: pd.DataFrame,
        athletes_df: pd.DataFrame,
        hosts_df: pd.DataFrame,
        noc: str,
        current_year: int
) -> Dict:
    """计算指定国家在指定年份的特征(只使用该年之前的数据)"""

    # 选择该国历史数据(到上一届)
    historical_data = medal_counts_df[
        (medal_counts_df['NOC_Code'] == noc) &
        (medal_counts_df['Year'] < current_year)
        ].sort_values('Year')

    # 如果没有历史数据,返回默认值
    if len(historical_data) == 0:
        return get_default_features()

    features = {}

    # 取出最近 3 届数据（不足 3 行则取全部）
    last_3_games = historical_data.tail(min(len(historical_data), 3))
    # 计算金牌和总奖牌的均值
    features['gold_3_games_avg'] = last_3_games['Gold'].mean() if len(last_3_games) > 0 else 0
    features['total_3_games_avg'] = last_3_games['Total'].mean() if len(last_3_games) > 0 else 0
    # 计算金牌和总奖牌的标准差（至少需要 2 行数据）
    features['gold_std'] = last_3_games['Gold'].std() if len(last_3_games) > 1 else 0
    features['total_std'] = last_3_games['Total'].std() if len(last_3_games) > 1 else 0

    # 2. 增长率特征
    if len(historical_data) >= 2:
        last_gold = historical_data.iloc[-1]['Gold']
        prev_gold = historical_data.iloc[-2]['Gold']
        features['gold_growth_noc_rate'] = (last_gold - prev_gold) / prev_gold if prev_gold > 0 else 0

        last_total = historical_data.iloc[-1]['Total']
        prev_total = historical_data.iloc[-2]['Total']
        features['metal_growth_noc_rate'] = (last_total - prev_total) / prev_total if prev_total > 0 else 0

        # 历史增长率
        gold_changes = historical_data['Gold'].pct_change()
        total_changes = historical_data['Total'].pct_change()
        features['gold_growth_game_rate'] = gold_changes.mean()
        features['metal_growth_game_rate'] = total_changes.mean()
    else:
        features['gold_growth_noc_rate'] = 0
        features['metal_growth_noc_rate'] = 0
        features['gold_growth_game_rate'] = 0
        features['metal_growth_game_rate'] = 0

    # 3. 上一届成绩
    features['gold_num'] = historical_data.iloc[-1]['Gold']
    features['metal_num'] = historical_data.iloc[-1]['Total']

    # 4. 排名相关特征
    features['best_rank_last_3'] = last_3_games['Rank'].min()
    features['rank_trend'] = last_3_games['Rank'].mean()

    # 5. 参赛历史特征
    features['participation_times'] = len(historical_data)

    # 6. 历史最佳成绩
    features['best_gold'] = historical_data['Gold'].max()
    features['best_total'] = historical_data['Total'].max()

    # 7. 运动员相关特征
    athlete_data = athletes_df[
        (athletes_df['NOC'] == noc) &
        (athletes_df['Year'] < current_year)
        ]

    if len(athlete_data) > 0:
        # 上一届运动员特征
        last_year = athlete_data['Year'].max()
        last_year_athletes = athlete_data[athlete_data['Year'] == last_year]

        features['athletes_count'] = len(last_year_athletes['Name'].unique())

        # 老将比例
        previous_athletes = set(athlete_data[athlete_data['Year'] < last_year]['Name'])
        current_athletes = set(last_year_athletes['Name'])
        veteran_count = len(current_athletes.intersection(previous_athletes))
        features['veteran_ratio'] = veteran_count / len(current_athletes) if len(current_athletes) > 0 else 0

        # 明星运动员
        medal_counts = athlete_data[athlete_data['Medal']!="No medal"].groupby('Name').size()
        features['star_athlete_num'] = len(medal_counts[medal_counts >= 3])

        # 运动员增长率
        if len(athlete_data['Year'].unique()) > 1:
            yearly_counts = athlete_data.groupby('Year')['Name'].nunique()
            features['athletes_growth_rate'] = yearly_counts.pct_change().mean()
        else:
            features['athletes_growth_rate'] = 0

        # 运动员效率
        total_medals = len(athlete_data[athlete_data['Medal']!="No medal"])
        total_gold = len(athlete_data[athlete_data['Medal'] == 'Gold'])
        total_athletes = len(athlete_data['Name'].unique())
        features['medals_per_athlete'] = total_medals / total_athletes if total_athletes > 0 else 0
        features['gold_per_athlete'] = total_gold / total_athletes if total_athletes > 0 else 0

    # 8. 项目相关特征
        last_year_events = athlete_data[athlete_data['Year'] == last_year]
        features['unique_sports_last'] = last_year_events['Sport'].nunique()
        features['unique_events_last'] = last_year_events['Event'].nunique()

        # 项目增长率
        yearly_events = athlete_data.groupby('Year')['Event'].nunique()
        yearly_sports = athlete_data.groupby('Year')['Sport'].nunique()
        if len(yearly_events) > 1:
            features['event_growth'] = yearly_events.pct_change().mean()
            features['sport_growth'] = yearly_sports.pct_change().mean()
        else:
            features['event_growth'] = 0
            features['sport_growth'] = 0

        # 大项集中度
        sport_medals = athlete_data[athlete_data['Medal']!="No medal"].groupby('Sport').size()
        total_medals = sport_medals.sum()
        if total_medals > 0:
            sport_shares = sport_medals / total_medals
            features['hhi_sport'] = (sport_shares ** 2).sum()
        else:
            features['hhi_sport'] = 0

        # 优势大项数目
        if total_medals > 0:
            features['dominant_sport'] = len(sport_shares[sport_shares > 0.5])
        else:
            features['dominant_sport'] = 0

        # 小项目效率
        medal_events = last_year_events[last_year_events['Medal']!="No medal"]['Event'].nunique()
        gold_events = last_year_events[last_year_events['Medal'] == 'Gold']['Event'].nunique()
        total_events = last_year_events['Event'].nunique()

        features['gold_event_rate'] = gold_events / total_events if total_events > 0 else 0
        features['metal_event_rate'] = medal_events / total_events if total_events > 0 else 0

        # 历史小项目效率:平均每个项目的金牌数目和奖牌数目
        event_gold_counts = athlete_data[athlete_data['Medal'] == 'Gold'].groupby('Event').size()
        event_medal_counts = athlete_data[athlete_data['Medal'] != 'No medal'].groupby('Event').size()
        total_participated_events = athlete_data['Event'].nunique()

        features[
            'gold_event_avg'] = event_gold_counts.sum() / total_participated_events if total_participated_events > 0 else 0
        features[
            'metal_event_avg'] = event_medal_counts.sum() / total_participated_events if total_participated_events > 0 else 0

        # 计算连续参加3届或以上的项目数
        sport_years = athlete_data.groupby('Sport')['Year'].apply(sorted)  # 按年份排序
        consistent_sports_count = 0
        for sport, years in sport_years.items():
            # 计算连续年份的最大长度
            max_consecutive = 1
            current_streak = 1
            for i in range(1, len(years)):
                if years[i] - years[i - 1] == 4:  # 判断是否连续
                    current_streak += 1
                    max_consecutive = max(max_consecutive, current_streak)
                else:
                    current_streak = 1  # 重置连续计数
            # 如果该项目连续参加了超过3届
            if max_consecutive >= 3:
                consistent_sports_count += 1
        features['consistent_sports'] = consistent_sports_count
        # 11. 距离上次获奖的年份差
        last_medal_year = historical_data[historical_data['Total'] > 0]['Year'].max()
        features['years_since_last_medal'] = current_year - last_medal_year if pd.notna(last_medal_year) else 400
    hosts_data = hosts_df[hosts_df['Year'] < current_year]
    features['is_host_history'] = int(any(hosts_data['Host'].str.contains(noc)))
    return features


def get_default_features():
    """返回特征的默认值"""
    return {
        'gold_3_games_avg': 0,
        'total_3_games_avg': 0,
        'gold_growth_noc_rate': 0,
        'metal_growth_noc_rate': 0,
        'gold_growth_game_rate': 0,
        'metal_growth_game_rate': 0,
        'gold_num': 0,
        'metal_num': 0,
        'best_rank_last_3': 0,
        'rank_trend': -1,
        'unique_sports_last': 0,
        'unique_events_last': 0,
        'event_growth': 0,
        'sport_growth': 0,
        'hhi_sport': 0,
        'dominant_sport': 0,
        'is_host_history': 0,
        'gold_event_rate': 0,
        'metal_event_rate': 0,
        'gold_event_avg': 0,
        'metal_event_avg': 0,
        'gold_std': 0,
        'total_std': 0,
        'athletes_count': 0,
        'athletes_growth_rate': 0,
        'veteran_ratio': 0,
        'star_athlete_num': 0,
        'medals_per_athlete': 0,
        'gold_per_athlete': 0,
        'years_since_last_medal': 0,
        'best_gold': 0,
        'best_total': 0,
        'participation_times': 0,
        'consistent_sports': 0
    }


def create_lgb_dataset(
        medal_counts_df: pd.DataFrame,
        athletes_df: pd.DataFrame,
        hosts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    创建用于LightGBM的数据集
    """
    dataset = []
    years = sorted(medal_counts_df['Year'].unique())

    for year in years[1:]:  # 从第二届开始,因为需要历史数据
        # 获取该届参赛的所有国家
        nocs = medal_counts_df[medal_counts_df['Year'] == year]['NOC_Code'].unique()

        # 计算历史东道主的平均增长率
        hosts_data = hosts_df[hosts_df['Year'] < year]  # 获取当前年份之前的所有东道主数据
        host_countries = hosts_data['Host'].unique()  # 所有历史东道主国家
        gold_growth_rates = []  # 存储金牌增长率
        total_growth_rates = []  # 存储总奖牌增长率
        for host_country in host_countries:
            # 获取该国家成为东道主的年份
            host_years = hosts_data[hosts_data['Host'] == host_country]['Year']
            for host_year in host_years:
                # 当前东道主年份的表现
                host_performance = medal_counts_df[
                    (medal_counts_df['Year'] == host_year) &
                    (medal_counts_df['NOC'] == host_country)
                    ]
                # 上一届的表现
                prev_performance = medal_counts_df[
                    (medal_counts_df['Year'] == host_year - 4) &
                    (medal_counts_df['NOC'] == host_country)
                    ]

                if len(host_performance) > 0 and len(prev_performance) > 0:
                    # 金牌增长率
                    prev_gold = prev_performance.iloc[0]['Gold']
                    host_gold = host_performance.iloc[0]['Gold']
                    gold_growth = (host_gold - prev_gold) / prev_gold if prev_gold > 0 else 0
                    gold_growth_rates.append(gold_growth)

                    # 总奖牌增长率
                    prev_total = prev_performance.iloc[0]['Total']
                    host_total = host_performance.iloc[0]['Total']
                    total_growth = (host_total - prev_total) / prev_total if prev_total > 0 else 0
                    total_growth_rates.append(total_growth)
        # 计算平均增长率
        avg_host_gold_growth = np.mean(gold_growth_rates) if gold_growth_rates else 0
        avg_host_total_growth = np.mean(total_growth_rates) if total_growth_rates else 0
        for noc in nocs:
            # 计算特征
            features = calculate_features_for_lgb(
                medal_counts_df, athletes_df, hosts_df, noc, year
            )
            filtered_df = medal_counts_df[(medal_counts_df["NOC_Code"] == noc) & (medal_counts_df["Year"] == year)]
            if not filtered_df.empty:
                is_host = filtered_df['is_host'].iloc[0]
            else:
                is_host = 0  # 默认值
            ishost_hostrate_gold =  avg_host_gold_growth * is_host
            ishost_hostrate_metal =  avg_host_total_growth * is_host
            # 添加目标变量
            current_result = medal_counts_df[
                (medal_counts_df['Year'] == year) &
                (medal_counts_df['NOC_Code'] == noc)
                ].iloc[0]

            features.update({
                'NOC_Code': noc,
                'Year': year,
                'gold_predict': current_result['Gold'],
                'metal_predict': current_result['Total'],
                'ishost_hostrate_gold': ishost_hostrate_gold,
                'ishost_hostrate_metal': ishost_hostrate_metal,
                'ishost':is_host
            })

            dataset.append(features)

    return pd.DataFrame(dataset)


def split_data_for_lgb(time_varying_df):
    """
    划分训练集和测试集
    训练集：2020年之前的所有数据
    测试集：2020-2024年间获得首枚奖牌的国家的数据
    """
    # 获取训练集：2020年之前的所有数据
    train_data = time_varying_df[time_varying_df['Year'] < 2020]

    # 获取这些国家的所有历史数据作为测试集
    test_data = time_varying_df[
        (time_varying_df['Year'].isin([2020, 2024]))
        ]

    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 检查数据集
    print("\n训练集年份范围:", train_data['Year'].min(), "-", train_data['Year'].max())
    print("测试集年份范围:", test_data['Year'].min(), "-", test_data['Year'].max())
    return train_data, test_data


def main():
    # 读取数据
    medal_counts = pd.read_csv("../summerOly_medal_counts.csv")
    athletes = pd.read_csv("../summerOly_athletes.csv")
    hosts = pd.read_csv("../summerOly_hosts.csv")

    # 数据预处理(添加NOC_Code等)
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
    unmapped_nocs = medal_counts[pd.isna(medal_counts['NOC_Code'])]
    if len(unmapped_nocs) > 0:
        print("Warning: Unmapped NOCs detected!")
        print(unmapped_nocs)
    # 国家名称映射（处理不一致问题）
    country_name_map = {
        "United Kingdom": "Great Britain",
        "Japan (postponed to 2021 due to the coronavirus pandemic)": "Japan"
    }
    # 清理字符串中的非标准空格和多余空格
    hosts["Host"] = hosts["Host"].str.replace(r'\xa0', ' ', regex=True).str.strip()
    # 提取主办国家并应用名称映射
    hosts["host_country"] = hosts["Host"].apply(lambda x: x.split(",")[-1].strip())
    hosts["host_country"] = hosts["host_country"].replace(country_name_map)
    # 合并主办国信息到 medal_counts
    medal_counts = pd.merge(medal_counts, hosts[["Year", "host_country"]], on="Year", how="left")
    # 判断是否为主办国
    medal_counts["is_host"] = (medal_counts["NOC"] == medal_counts["host_country"]).astype(int)

    # 创建数据集
    lgb_dataset = create_lgb_dataset(medal_counts, athletes, hosts)
    lgb_dataset.dropna(inplace=True)
    train_data,test_data = split_data_for_lgb(lgb_dataset)
    # 保存数据集
    lgb_dataset.to_csv("./lgb_dataset.csv", index=False)
    train_data.to_csv("./lgb_train_data.csv", index=False)
    test_data.to_csv("./lgb_test_data.csv", index=False)
    # print(lgb_dataset.isna().sum())
    print(lgb_dataset.columns)
    lgb_predict_dataset = build_prediction_dataset_2028(medal_counts,athletes,hosts)
    print(lgb_predict_dataset.isna().sum())
    print(lgb_predict_dataset.head())
    lgb_predict_dataset.to_csv("./lgb_predict_dataset.csv",index=False)


def build_prediction_dataset_2028(medal_counts_df: pd.DataFrame, athletes_df: pd.DataFrame, hosts_df: pd.DataFrame):
    """
    构建2028年预测数据集

    Parameters:
    -----------
    medal_counts_df : DataFrame
        奖牌数据集
    athletes_df : DataFrame
        运动员数据集
    hosts_df : DataFrame
        主办国数据集
    """
    # 获取2024年获得奖牌的国家
    countries_2024 = medal_counts_df[
        (medal_counts_df['Year'] == 2024) &
        (medal_counts_df['Total'] > 0)
        ]['NOC_Code'].unique()

    # 构建2028年的预测数据
    predict_data = []
    for noc in countries_2024:
        # 计算特征
        features = calculate_features_for_lgb(
            medal_counts_df, athletes_df, hosts_df, noc, 2028
        )

        # 添加2028年是否为主办国信息
        is_host = 1 if noc == 'USA' else 0

        # 计算主办国效应
        host_years = hosts_df['Year'].unique()
        host_countries = hosts_df['Host'].unique()

        gold_growth_rates = []
        total_growth_rates = []

        # 计算历史上主办国的平均增长率
        for host_year, host_country in zip(host_years, host_countries):

            host_performance = medal_counts_df[
                (medal_counts_df['Year'] == host_year) &
                (medal_counts_df['NOC'] == host_country)
                ]
            prev_performance = medal_counts_df[
                (medal_counts_df['Year'] == host_year - 4) &
                (medal_counts_df['NOC'] == host_country)
                ]

            if len(host_performance) > 0 and len(prev_performance) > 0:
                # 金牌增长率
                prev_gold = prev_performance.iloc[0]['Gold']
                host_gold = host_performance.iloc[0]['Gold']
                if prev_gold > 0:
                    gold_growth_rates.append((host_gold - prev_gold) / prev_gold)

                # 总奖牌增长率
                prev_total = prev_performance.iloc[0]['Total']
                host_total = host_performance.iloc[0]['Total']
                if prev_total > 0:
                    total_growth_rates.append((host_total - prev_total) / prev_total)

        # 计算平均主办国效应
        avg_host_gold_growth = np.mean(gold_growth_rates) if gold_growth_rates else 0
        avg_host_total_growth = np.mean(total_growth_rates) if total_growth_rates else 0

        # 更新特征
        features.update({
            'NOC_Code': noc,
            'Year': 2028,
            'ishost': is_host,
            'ishost_hostrate_gold': avg_host_gold_growth * is_host,
            'ishost_hostrate_metal': avg_host_total_growth * is_host
        })

        predict_data.append(features)

    predict_df = pd.DataFrame(predict_data)
    print(f"\nNumber of countries to predict: {len(predict_df)}")
    print(f"Features in prediction dataset: {predict_df.columns.tolist()}")

    return predict_df

if __name__ == "__main__":
    main()