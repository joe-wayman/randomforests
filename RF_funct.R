#' Code used to run Random Forests for the BTO temporal paper. 
#' 
#' 
#' @description This function trains and tunes a randomforest, and then evaluates that forest on 10 seperate training and tesing folds. The
#' function uses 80% of the data for training and 20% for testing. The tuning aspect runs the random forests with 
#' different values of mtry (number of variables used in each split) and n (the number of forests made) until there is 
#' no longer a 0.01 increase in R2.
#' @param data = the data frame to use
#' @param CP = The number of cores to use in the parallel processing
#' @param Response = The response variable name
#' @param CoVars = The vector of predictor variable names
#' @param dirSave = The directory where the models are saved. Should be in the format "location/RF/"
#' @param scale = The scale of the study (blank by default, BTO specific)
#' @output A model saved in the directory location. The model output is a list of the 10 models, each trained and tested
#' with a separate block of data. 
#' @return Same as above. 


RF_funct <- function(data, CP, Response, CoVars, dirSave, scale = ""){
  
  require(dplyr)
  require(caret)
  require(snowfall)
  
  
  inTrain <- createFolds(data[,Response], k = 10, list = T)
  
  blocks_2 <- data.frame()
  
  for(i in 1:length(inTrain)){
    
    block <- data[inTrain[[i]],]
    block$block <- i
    blocks_2 <- rbind(blocks_2, block)
    
}
  
  data.model <- blocks_2
  
  predNames <- sapply(c(CoVars),function(x,...) colnames(data.model)[which(colnames(data.model)==x)]) # get predictor names
  
  formula = as.formula(paste(Response, "~" , paste(CoVars, collapse = "+"), sep = ""))

  ##Set up parallel
  
  sfInit(parallel=TRUE, cpus=CP)
  
  sfLibrary(randomForest)#;sfLibrary(PresenceAbsence); sfLibrary(modEvA)
  
  sfExport(list=c("data.model","formula", "inTrain", "Response"))
  
  
  
  
  # Tune the models using different values of mtry and n
  
  
  mod <- lapply(1:10, function(m){

    mod <- sfLapply(1:length(inTrain), function(blockNo){
      
      
      cat('\n',"block = ",blockNo,"; mtry = ",m,'\n',sep=" ")
      
      fit.blocks <- subset(data.model, block != blockNo)
      
      test.block <- subset(data.model, block == blockNo)
      
      init.ntree <- 1000; init.Rsq <- 0.01; mod.improve <- "TRUE"
      
    
      while(mod.improve == "TRUE"){   
        
        
        
        model1 <- randomForest(formula, data = fit.blocks, ntree = init.ntree, mtry = m)   
        
        PRED <- predict(model1,newdata=test.block,type="response",se.fit=FALSE)
        
        PRED <- as.data.frame(PRED)
        
        test.block$id<-row.names(test.block)
        
        eval.data.rsq <- cbind(test.block[,c("id",Response)],PRED)
        
        
        
        ## Evaluate model fit
        
        SSE<-sum((eval.data.rsq[,Response]-eval.data.rsq$PRED)^2, na.rm=T)
        
        TSS<-sum((eval.data.rsq[,Response]-mean(fit.blocks[,Response]))^2, na.rm = T)
        
        Rsq<-1-(SSE/TSS)
        
        
        
        try(if(((Rsq/init.Rsq)) > 1.01){mod.improve <- "TRUE";init.Rsq <- Rsq;init.ntree <- init.ntree +500}else{mod.improve <- "FALSE"; best.ntree<-(init.ntree-500)}, silent=T)
        
        if(is.na(Rsq)){mod.improve <- "FALSE"; best.ntree<-(init.ntree-500); Rsq<-init.Rsq}
        
      }  
      
      
      return(list(block = blockNo, mtry=m, ntree=best.ntree, Rsq = init.Rsq))
      
      cat("\n")
      
    })
    
    return(mod)
    
})

  
  
  
  
  
  ##Find optimum values
  
  evaluation.tab <- as.data.frame(matrix(unlist(mod),ncol=4,byrow=TRUE)); colnames(evaluation.tab) <- c("block","mtry","ntrees","Rsq")
  
  Rsq.mean.mtry <- aggregate(Rsq~mtry,FUN=mean,data=evaluation.tab)
  
  opt.mtry <-  Rsq.mean.mtry[which(Rsq.mean.mtry[,2]==max(Rsq.mean.mtry[,2])),1]
  
  opt.ntree <- max(evaluation.tab$ntrees)
  
  
  
  ##Set up parallel
  
  sfExport(list=c("data.model","formula","opt.mtry","opt.ntree","Response"))
  
  
  
  #fit final models to data

  mod <- sfLapply(1:length(inTrain),function(blockNo,m,nt){
    
    
    
    cat('\n', "Fitting final model to block",blockNo,'\n',sep=" ")
    
    fit.blocks <- subset(data.model, block != blockNo)
    
    test.block <- subset(data.model, block == blockNo)
    
    
    
    model1 <- randomForest(formula, data = fit.blocks, ntree = nt, mtry=m, importance = T)
    
    PRED <- predict(model1,newdata=test.block,type="response",se.fit=FALSE)
    
    PRED <- as.data.frame(PRED)
    
    test.block$id<-row.names(test.block)
    
    eval.data.Rsq <- cbind(test.block[,c("id",Response)],PRED)
    
    eval.data.Rsq<-eval.data.Rsq[complete.cases(eval.data.Rsq),]
    
    
    
    SSE<-sum((eval.data.Rsq[,Response]-eval.data.Rsq$PRED)^2)
    
    TSS<-sum((eval.data.Rsq[,Response]-mean(fit.blocks[,Response]))^2)
    
    Rsq<-1-(SSE/TSS)
    
    
    
    return(list(block=blockNo, Rsq=Rsq, mod=model1, mtry=m, ntree=nt))
    
  },m = opt.mtry, nt = opt.ntree)  # 2 minutes
 

  
  
  
  ##Save block models
  save(mod, file=paste(dirSave, Response, "_", scale, "_model.output.RF.rda",sep=""), compress = "bzip2")

  

  
  sfStop()
  
  return(mod)
  
  

}





































