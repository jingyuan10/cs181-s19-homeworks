#Clean global environment
rm(list=ls())
gc()
setwd("/Users/jingyuan/Dropbox/P4_Jing")

#Library
#Library#
library(tidyverse)
library(dplyr)
library(plyr)
library(Matrix)
library(tidyr)
library(ggplot2)
library(gamm4)
library(lmerTest)
library(splitstackshape)
library(reshape2)
library(tidyr)
library(data.table)
library(weights)
library(Rmisc)

# Read excel data
alpha=read.csv("test_alpha.csv")
alpha.dat=data.frame(alpha)
alpha.dat$alpha=as.factor(alpha.dat$alpha)

gamma=read.csv("test_gamma.csv")
gamma.dat=data.frame(gamma)
gamma.dat$gamma=as.factor(gamma.dat$gamma)


q2=read.csv("test_Q2.csv")
q2.dat=data.frame(q2)
q1=read.csv("test_Q1.csv")
q1.dat=data.frame(q1)

model2=ddply(q1.dat,.(alpha,gamma,epoch,highest,average,score), summarise,
             model=2)
model2.1=ddply(q2.dat,.(alpha,gamma,epoch,highest,average,score), summarise,
            model=2.1)
model1=ddply(filter(gamma.dat, gamma==0.6),.(alpha,gamma,epoch,highest,average,score), summarise,
                 model=1)
model_compare=merge(x=merge(x=model1,y=model2,all=T),y=model2.1,all=T)
model_compare$model=as.factor(model_compare$model)

gravity=read.csv("test_gravity.csv")
gravity.dat=data.frame(gravity)
gravity.dat=ddply(gravity.dat,.(alpha,gamma,epoch,highest,average,score), summarise,
             model=3)
gravity.dat.m=merge(x=merge(x=model1,y=gravity.dat,all=T),y=model2.1,all=T)
gravity.dat.m$model=as.factor(gravity.dat.m$model)

#Plot
ggplot()+
  ggtitle("Average Score with different learing rate") +
  geom_line(data=alpha.dat, aes(x=epoch, y=average,group=alpha,color=alpha),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Higehst Score with different learing rate")+ 
  geom_line(data=alpha.dat, aes(x=epoch, y=highest,group=alpha,color=alpha),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))
  

ggplot()+
  ggtitle("Score with different learing rate")+ 
  geom_line(data=alpha.dat, aes(x=epoch, y=score,group=alpha,color=alpha),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))


#Plot
ggplot()+
  ggtitle("Average Score with different discount rate") +
  geom_line(data=gamma.dat, aes(x=epoch, y=average,group=gamma,color=gamma),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Higehst Score with different discount rate")+ 
  geom_line(data=gamma.dat, aes(x=epoch, y=highest,group=gamma,color=gamma),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))


ggplot()+
  ggtitle("Score with different discount rate")+ 
  geom_line(data=gamma.dat, aes(x=epoch, y=score,group=gamma,color=gamma),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

#Plot
ggplot()+
  ggtitle("Average Score with different model") +
  geom_line(data=model_compare, aes(x=epoch, y=average,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Higehst Score with different model")+ 
  geom_line(data=model_compare, aes(x=epoch, y=highest,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Score with different model")+ 
  geom_line(data=model_compare, aes(x=epoch, y=score,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))
# consider gravity
ggplot()+
  ggtitle("Average Score with different model") +
  geom_line(data=gravity.dat.m, aes(x=epoch, y=average,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Higehst Score with different model")+ 
  geom_line(data=gravity.dat.m, aes(x=epoch, y=highest,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))

ggplot()+
  ggtitle("Score with different model")+ 
  geom_line(data=gravity.dat.m, aes(x=epoch, y=score,group=model,color=model),size=0.2)+
  xlab("")+
  ylab("")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor= element_blank(),
        plot.title=element_text(size=10))