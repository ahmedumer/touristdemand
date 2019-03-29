library("strucchange")
library("aTSA")
setwd('arrivals')

#----------check diff between international, regional and domesitc arrivals using anova----------#
df_ar <- read.table('arrivals.tsv', sep='\t', header = TRUE)

#anova for the three groups
dati <- c(df_ar$MSIQ, df_ar$ME, df_ar$Int)
groups <- factor(rep(letters[1:3], each = 16))
anova(lm(dati ~ groups))

#anova for diff of the three groups
dati_diff <- c(diff(df_ar$MSIQ, differences=1), diff(df_ar$ME, differences=1), diff(df_ar$Int, differences=1))
groups_diff <- factor(rep(letters[1:3], each = 15))
anova(lm(dati_diff ~ groups_diff))
#not possible to reject the null because p-value is insig therefore no diff


#---------------------Structural Break analysis and unit roots------------------#
df <- read.table('dataset_Jan2003Sep2018.csv', 
                 sep='\t', header=TRUE, row.names = 1)

df$date <- as.Date(as.character(df$date))

#check for unit root
adf.test(df$n)

#convert to time series
n_ts_unadj <- ts(data=df$n, frequency = 12, start=c(2003, 1,1), end=c(2018, 9,1))
n_2009 <- ts(data=df$n, frequency = 12, start=c(2003, 1,1), end=c(2009, 12,1))
nkill_ts <- ts(data=df$nkill, frequency = 12, start=c(2003, 1,1), end=c(2018, 9,1))

#seasonally adjustment
n_ts <- n_ts_unadj - decompose(n_ts_unadj)$seasonal
n_2009 <- n_2009 - decompose(n_2009)$seasonal

df['adj_n'] = n_ts

#---------Plot the number of fatalities-----------#
par(mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(3.3,3.6,1.1,1.1))
jpeg('num_fatalities.jpg', width = 900, height = 400, units = "px", quality = 900, pointsize=14)
plot(nkill_ts, xlab='Years', ylab='Number of fatalities (total per month)', type='l',
     xaxt="n")
axis(1, at=c(2003,2018), labels=c("",""), lwd.ticks=0)
axis(1, at=seq(2003 , 2018, by=1), lwd=0, lwd.ticks=0)
dev.off()

#--------Plot n arrivals, seasonality and decomposied n arrivals------------#
options(scipen=5)
jpeg('n_ts.jpg', width = 1400, height = 1000, units = "px", quality = 900, pointsize=18, res=100)
par(mfrow=c(3,1), mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(3.3,3.6,1.1,1.1))

plot(n_ts_unadj, xlab='', ylab='Un-adjusted', type='l',
     xaxt="n", yaxt='n')
#axis(1, at=c(2003,2018), labels=c("",""), lwd.ticks=0)
#axis(1, at=seq(2003 , 2018, by=1), lwd=0, lwd.ticks=1)
axis(2, at=c(0, 400000), labels=c("",""), lwd.ticks=0)
axis(2, at=seq(0 , 400000, by=100000), lwd=0, lwd.ticks=1)

plot(decompose(n_ts_unadj)$seasonal, xlab='', ylab='Seasonality', type='l',
     xaxt="n", yaxt='n')
#axis(1, at=c(2003,2018), labels=c("",""), lwd.ticks=0)
#axis(1, at=seq(2003 , 2018, by=1), lwd=0, lwd.ticks=1)
axis(2, at=c(-35000, 25000), labels=c("",""), lwd.ticks=0)
axis(2, at=seq(-35000 , 25000, by=10000), lwd=0, lwd.ticks=1)

plot(n_ts, xlab='Years', ylab='Adjusted', type='l',
     xaxt="n", yaxt='n')
axis(1, at=c(2003,2018), labels=c("",""), lwd.ticks=0)
axis(1, at=seq(2003 , 2018, by=1), lwd=0, lwd.ticks=1)
axis(2, at=c(0, 400000), labels=c("",""), lwd.ticks=0)
axis(2, at=seq(0 , 400000, by=100000), lwd=0, lwd.ticks=1)
dev.off()

#--------Plot Differences in number of arrivals------------#
options(scipen=1)
jpeg('difflag1.jpg', width = 1400, height = 1000, units = "px", res=100, pointsize=18)
par(mfrow=c(2,2), mgp=c(2.2,0.45,0), tcl=-0.4, mar=c(3.3,3.6,1.1,1.1))
plot(diff(n_ts, differences=1, trim=TRUE), ylab="Differences=1", xaxt="n", xlab="")
plot(diff(n_ts, differences=12, trim=TRUE), ylab="Differences=12", xaxt="n", xlab="")
plot(diff(n_ts, differences=24), ylab="Differences=24", xlab="Years")
plot(diff(n_ts, differences=36), ylab="Difference=36", xlab="Years")
dev.off()

par(mfrow=c(1,1))

#transform n, exp(n_ts)
nl_ts <- n_ts^2 #ts(data=df$ln, frequency = 12, start=c(2003, 1, 1), end=c(2018, 9, 1))
nl_2009 <- n_2009^2


#--------Break points analysis------------#
#get break points in the whole period and plot them
n_ts_brk <- breakpoints(nl_ts ~ 1, h = 0.1)
summary(n_ts_brk)
breakdates(n_ts_brk, breaks = 4)
coef(n_ts_brk, breaks = 4)

#get break points only between 2003 and 2009
n_ts_brk_2009 <- breakpoints(nl_2009 ~  1, h = 0.1)
summary(n_ts_brk_2009)
breakdates(n_ts_brk_2009, breaks = 3)
coef(n_ts_brk_2009, breaks = 3)

jpeg('breakpoints_BICRSS.jpg', width = 1400, height = 500, units = "px", pointsize=14)
par(mfrow=c(1,2))
plot(n_ts_brk, main=" 2003-2018")
plot(n_ts_brk_2009, main="2003-2009")
dev.off()

#--------Plot Break dates and number of arrivals------------#
#plot with break dates
options(scipen=4)
#plot the data with the identified breaks
jpeg('breakdates_2018.jpg', width = 1000, height = 500, units = "px", quality = 900, pointsize=14)
#par(mfrow=c(3,1))
plot(nl_ts, xlab='Years', ylab='Number of arrivals (SQ)', type='l',
     xaxt="n") #, yaxt='n')
axis(1, at=c(2003,2018), labels=c("",""), lwd.ticks=0)
axis(1, at=seq(2003 , 2018, by=1), lwd=0, lwd.ticks=1)
lines(fitted(n_ts_brk, breaks = 4), col = 4)
lines(confint(n_ts_brk, level=.95, breaks = 4))
dev.off()

jpeg('breakdates_2009.jpg', width = 1000, height = 500, units = "px", quality = 900, pointsize=14)
plot(nl_2009, xlab='Years', ylab='Number of arrivals (SQ)', type='l',
     xaxt="n") #, yaxt='n')
axis(1, at=c(2003,2010), labels=c("",""), lwd.ticks=0)
axis(1, at=seq(2003 , 2010, by=1), lwd=0, lwd.ticks=1)
lines(fitted(n_ts_brk_2009, breaks =3), col = 4)
lines(confint(n_ts_brk_2009, breaks = 3))
dev.off()

#get F-stat for the 2007 break point
n_7 <- (ts(data=df$n, frequency = 12, start=c(2003, 1,1), end=c(2007, 12,1)))^2
plot(Fstats(n_7 ~ 1))
lines(breakpoints(Fstats(n_7 ~ 1)))
sctest(Fstats(n_7 ~ 1))
