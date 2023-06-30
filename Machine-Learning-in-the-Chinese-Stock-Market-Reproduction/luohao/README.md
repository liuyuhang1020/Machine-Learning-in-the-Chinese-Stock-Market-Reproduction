# Machine learning in the Chinese stock market



YYYY-MM-01描述MM的数据，不是指月初能获取的数据

or a specific model, we calculate the reduction in predictive R2 when setting all values of a given predictor to zero within each training sample, and average them into a single importance measure for each predictor.

for a specific model, we calculate the reduction in predictive R2 when setting all values of a given predictor to zero within each training sample, and average them into a single importance measure for each predictor.

## 重要变量

3.4. Dissecting the predictability performance of NN4

宏观：

- ntis，the level of issuance activity
- infl
- m2gr
- itgr

微观：market liquidity， fundamental signals and valuation ratios， risk measures

- std_dolvol：volatility of liquidity

- std_turn：同上

- zerotrade：zero trading days

- atr

- er_trend

- chaotia：industry-adjusted change in asset turnover

- ill：illiquidity measure

- chempia：industry-adjusted change in employees

- mve：total market value

- nincr：number of recent earning increases

- chpmia：industry-adjusted change in profit margin

- bm_ia：industry-adjusted book-to-market

- egr

- orgcap

- maxret

- sgr

- pchgm_pchsale

- rsup

- sp

- turn

- ---

- idiovol：idiosyncratic return volatility

- volatility：total return volatility

- beta：market beta: BETA_Mbeta->Betavals 综合市场月Beta值

- atr：abnormal turnover ratio 

- top10holderrate：top-10 shareholders ownership



Alternative model selection

- USPA：unconditional superior pre- dictive ability
- CSPA：conditional superior predictive ability



# 服务器

![image-20221013203251609](笔记.assets/image-20221013203251609.png)

 ssh bigboy@29g28162v8.qicp.vip

wifi： WirelessWict，密码为：Wict2020@zGc128

lhao@172.31.203.252

icst123

scp -r data/日交易统计文件 lhao@172.31.203.252:stock/data



scp  lhao@29g38162v8.qicp.vip:../../S1/luohao/stock/data/trend/tmp5.csv data_er



scp 表格格式转换.py lhao@172.31.203.252:stock



scp  -P data/日交易统计文件/daily_info.csv lhao@29g38162v8.qicp.vip: /stock/data

scp  -P 22006 api.py lhao@29g38162v8.qicp.vip: /stock



source activate pytorch 进入pytorch环境



## 表格与csmar名称对应关系

日度

- 股本变动文件 TRD_Capchg 1.8显示1.16
- 日个股回报率 1.8显示1.6 
- 分配文件 TRD_Cptl 1.8显示1.11 要买
- 公司公告事件表 1.8显示1.7 要买
- 日交易统计文件 LT_Dailyinfo 12.30 要买

月度

- 月个股回报率文件 TRD_Mnth 12月 
- beta BETA_Mbeta 12月 要买 

季度

- 资产负债表 FS_Combas 9.30
- 利润表 FS_Comins 9.30
- 现金流量表(直接法) FS_Comscfd 9.30
- 现金流量表(间接法) FS_Comscfi 9.30
- 相对价值指标 FI_T10 9.30

年度

- 年中季报事件表 ER_RelForcDate 12.31 
- 治理综合信息文件 CG_Ybasic 23年1月显示21.12.31 



## 因子更新情况

23.1.12

原始数据更新情况：

日度

- 股本变动文件 TRD_Capchg 1.8显示1.16
- 日个股回报率 1.8显示1.6 
- 分配文件 TRD_Cptl 1.8显示1.11 要买
- 公司公告事件表 1.8显示1.7 要买
- 日交易统计文件 LT_Dailyinfo 12.30 要买

月度

- 月个股回报率文件 TRD_Mnth 12月 
- beta BETA_Mbeta 12月 要买 

季度

- 资产负债表 FS_Combas 9.30
- 利润表 FS_Comins 9.30
- 现金流量表(直接法) FS_Comscfd 9.30
- 现金流量表(间接法) FS_Comscfi 9.30
- 相对价值指标 FI_T10 9.30

年度

- 年中季报事件表 ER_RelForcDate 12.31 
- 治理综合信息文件 CG_Ybasic 23年1月显示21.12.31 

因子更新情况：

- 未更新
  - 1 absacc
  - 2 acc
  - 3 agr
  - 6 bm
  - 7 bm_ia
  - 8 cash
  - 9 cashdebt
  - 10 cashspr
  - 11 cfp
  - 12 cfp_ia
  - 13 chato 
  - 14 chato_ia
  - 16 chempia
  - 17 chinv
  - 28 dy
  - 30 egr
  - 31 gma
  - 32 grCAPX
  - 33 herf
  - 34 hire
  - 37 invest
  - 38 lev
  - 39 lgr
  - 46 mve
  - 47 mve_ia
  - 48 nincr
  - 50 orgcap
  - 51 pchcapx_ia
  - 60 pctacc
  - 71 roeq
  - 72 roic
  - 81 stdacc
  - 83 tang
- 更新
  - 4 beta（统一使用csmar上的月beta数据，和之前的有较大不同。5同）
  - 5 betasq
  - 15 chcsho
  - 18 chmom
  - 25 divi 
  - 26 divo 
  - 29 ear 盈利公告前后共3天的日回报率之和。不是定期公布的，需要注意
  - 35 idiovol 
  - 36 ill 
  - 40 maxret
  - 41 mom12m
  - 42 mom1m
  - 43 mom6m
  - 44 mom36m
  - 88 er_trend 

更新的因子时间范围是199012-202301。202301只是占位，实际没有填充。



#### 因子更新  2023.2

所有月度频率因子列举如下：

- 04 beta，05 betaasq（罗），15 chcsho（罗），18 chmom（罗）

  - 定义不变，不修改

- 27 dolvol
  - 定义不变，不修改

  - dolvol ok 已修改,由t-2月改为t-1月

- 35idiovol，36ill（罗）

  - 定义不变，不修改

- 40-44 mom系列（maxret,mom1m,mom6m,mom12m,mom36m），68volatility，86zerotrade
  - **修改定义为结束于t月**，已修改（原数据全部左移一格）
    - maxret ok t月
    - mom1m ok 202206 下是 20220601-20220630 按我们的定义是t月
    - mom6m ok 202206 下是sum:mom1m(202201-202205, 5 month) 按我们的定义是t-5到t-1
    - mom12m ok 202206 下是sum:mom1m(202107-202205, 11 month) 按我们的定义是t-11到t-1
    - mom36m 
      - 现在 202206 下是sum:mom1m(201906-202105, 36 month)，
      - 存疑 新找的那篇论文中mom系列的定义我算的是 201907-202106，我理解的按我们的定义是t-35到t-12
      - 已更正
    - volatility(retvol) ok 按我们的定义 return volatility of month t
    - zerotrader ok t月末可以得到t月的数据，因此使用t月的数据


- 46mve，47mve_ia（罗）

  - 应当也要**修改定义为结束于t月**

- 79std_dolvol，80std_turn，85turn

  - 定义不变，不修改

  - std_dolvol 现在是t月的么
  - turn 由 t-3 ---> t-1 / t 变为 t-2 --> t / t
  - std_ turn 不用变

- 87atr，88er_trend（罗）

  - ok

- macro

  - 所有月度频率宏观因子定义均**不需修改**，但M_m2gr，M_itgr，M_infl，数据获取有延迟（1个月）
    - 此外，所有的都可以当月末获取么？

  

4.7

- beta √
- betasq √
- chcsho √
- er_trend √
- ill
