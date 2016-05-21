library(ggplot2)

df <- read.table("cutoff_dependence.txt", col.names = c('Cutoff', 'Integral', 'Error', 'Q'))

plt <- ggplot(data = df, aes(x=Cutoff, y=Integral))+
  geom_point()+
  geom_line()+
  geom_line(aes(y=Integral + 100 * Error))+
  geom_line(aes(y=Integral - 100 * Error))
plt

means = c(2.0422845507686063,
          2.080053985630342,
          2.0591208235765026,
          2.0800964574285334,
          2.0612631810399398,
          2.0081074276831306,
          2.0991958166820317,
          2.1256247376870356,
          2.096771639499009,
          1.986139083281772,
          2.0290075349535024,
          2.0375382847420136,
          2.049097548716797,
          2.0374626130702906,
          2.0097869695563455,
          2.0377474612266266,
          2.048508752191629,
          2.0247245532004223,
          2.0964223223534466,
          2.0586524628552496)

plt_hist = ggplot(mapping = aes(x=means))
plt_hist + geom_histogram(binwidth = 0.01)
