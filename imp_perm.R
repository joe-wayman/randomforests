## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##



## Extract Model performance and variable importance scores



## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


## Outpath = path where both the results are stored and the outputs are saved (format: "location/RF"). 

## scale = the scale which you want to calculate importance permutations for

## var = vector of explanatory variable names

## data = original data frame used to construct random forests

## CP = number of cores

## noperm = number of permutations per model. recommended = 100 (100 permutations
##          per model = 1000 over the 10 models)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

##        MODEL PROCESSING

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

importance_perm <- function(outpath, var, data, CP, noperm){
  
  
  require(snowfall)
  require(randomForest)
  require(dplyr)
  
  
  files <- list.files(outpath, pattern = ".rda", recursive = F)

  clim.var <- var
  
  my.data <- data
  
  sfInit(parallel=TRUE, cpus = CP)
  
  sfLibrary(randomForest)
  
  sfExport(list=c("files","my.data","outpath","clim.var"))
  
  sfLapply(files, function(f){
    
    
    ##~~~~~~~~  Load Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    
    load(paste(outpath, f, sep = "/"))
  
    ##~~~~~~~~ Get response variable name ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 
  
    res <- stringr::str_remove(f, pattern = paste0("__model.output.RF.rda")) 
    
    
    
    ##~~~~~~~~   Model performance and summary statistics~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    
    Rsq<-sapply(1:length(mod), function(i){Rsq<-mod[[i]]$Rsq})
    
    block<-seq(1,length(mod),1)
    
    perVarexp<-sapply(1:length(mod), function(i){perVarexp<-(mod[[i]]$mod$rsq[length(mod[[i]]$mod$rsq)])*100})
    
    mse<-sapply(1:length(mod), function(i){mse<-(mod[[i]]$mod$mse[length(mod[[i]]$mod$mse)])})
    
    nt<-sapply(1:length(mod), function(i){nt<-(mod[[i]]$mod$ntree)})
    
    mtr<-sapply(1:length(mod), function(i){nt<-(mod[[i]]$mod$mtry )})
    
    IntRsq<-sapply(1:length(mod), function(i){IntRsq<-mean(mod[[i]]$mod$rsq)})
    
    dat<-cbind(block, Rsq, perVarexp, mse, nt, mtr, IntRsq)
    
    write.csv(dat, paste(outpath, "/", res, "_", "Model_Performance.csv", sep = ""), row.names = F)  
    
    
    ##~~~~~~~~~~~~~~~Extract internal Variable Importance metrics for comparison ~~~~~~~~~~~~ ##
    
    VarImp<-lapply(1:length(mod), function(i){
      
      VI<-importance(mod[[i]]$mod, scale=F)
      
      VI<-as.data.frame(VI)
      
      VI$var<-row.names(VI)
      
      VI$block<-i
      
      return(VI)})
    
    VarImp<-Reduce(function(...) merge(..., all=T), VarImp)
    
    
    
    ##~~~~~~~~  Calculate variable importance using equation with bootstrapping ~~~~~~~~~~~~ ##
    
    
    
    CoVars<-unique(VarImp$var)
    
    
    MyDat <- my.data
    
    
    ## Manual Variable Importance
    
    VarImpMan <- lapply(1:length(mod), function(i){
      
      VIman <- do.call(rbind, lapply(CoVars, function(x){
        
        VImanrep <- do.call(rbind, lapply(1:noperm, function(v){
          
          dat <- MyDat 
          
          dat$PREDasis <- predict(mod[[i]]$mod, newdata = dat) # Prediction as the variable is
          
          dat$rand <- sample(dat[,x])                          # Sample the variable (randomise)
           
          dat[,x] <- dat$rand                                  # Store the random variable in the variable slot
          
          dat$PREDrand <- predict(mod[[i]]$mod, newdata=dat)   # Predict again using randomised variable
          
          MSEasis <- mean( (dat$PREDasis - dat[,res])^2, na.rm = TRUE)   # get the mean between 
          
          MSErand <- mean( (dat$PREDrand - dat[,res])^2, na.rm = TRUE)
          
          manVI <- sqrt((MSErand-MSEasis)/MSEasis)
          
          if(is.na(manVI)){manVI<-0} ## i.e. where mse using a randomised variable is < mse from variable as is. Variable Importance = 0
          
          return(c(x,MSEasis,MSErand, manVI))
          
        }))
        
        return(VImanrep)
        
      }))
      
      VIman<-as.data.frame(VIman, stringsAsFactors=F)
      
      colnames(VIman)<-c("var","MSEasis","MSErand","ManVI")
      
      VIman$block<-i
      
      return(VIman)
      
    })
    
    VarImpMan <- Reduce(function(...) merge(..., all=T), VarImpMan)
    
    colnames(VarImpMan)<-c("var","MSEasis","MSErand","ManVI","block")
    
    VI<-merge(VarImp, VarImpMan, by=c("var","block"))
    
    write.csv(VI, paste(outpath, "/", res, "_variable_importance_bootstrap.csv", sep = ""), row.names = F)
    
    
  })
  
}


