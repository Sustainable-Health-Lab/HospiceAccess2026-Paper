#                  INSTALL_opts = c("--no-lock","--no-multiarch",  "--no-test-load"))
# install.packages("marginaleffects", repos = NULL,
#                  contriburl = "file:/opt/software/cran/src/contrib/",
#                  type = "source",

# R.version.string

# install.packages("/opt/software/cran/src/contrib/lme4_1.1-37.tar.gz", repos=NULL, type="source")

library(Matrix)

library(lme4)


library(carData)
library(car)


library(dplyr)
library(purrr)
library(broom.mixed)
library(ggplot2)
library(margins)
library(grid)
library(ggeffects)
library(marginaleffects)

file_path <- "1-Data/Analysis_ContextPatientUtil.csv"

df <- read.csv(file_path)

print(colnames(df))


keep_col <- c("SEX", "ZIPCODE","agedgrp5", "RTI_RACE","STATE_x", "hospice_use",
              "ruca_category","Hispanic", "White","Black","Asian", "Black_pct","White_pct",
              "Hispanic_pct", "Asian_pct", "Income", "MIN_ToBreak", "Region", "race_str","is_asian",                     
              "is_black", "is_hispanic", "Suburban", "Urban", "is_female", "is15min", "is30min", "is60min")                       

df1 <- df %>% 
  select(keep_col)

str(df1)

df1$ZIPCODE <- as.factor(df1$ZIPCODE)

df1$Income_scaled <- scale(df1$Income)
df1$Income_log <- log(df1$Income, base=10)
df1$Income_log_scaled <- scale(df1$Income_log)



# df1$race_str <- factor(df1$race_str, levels = c("is_white", "is_black", "is_hispanic", "is_asian"))


# df1$MIN_ToBreak <- factor(df1$MIN_ToBreak, levels = c("is15min", "is30min", "is60min", "greaterthan60"))
df1$is15min <- ifelse(df1$MIN_ToBreak == "is15min", 1, 0)
df1$is30min <- ifelse(df1$MIN_ToBreak == "is30min", 1, 0)
df1$is60min <- ifelse(df1$MIN_ToBreak == "is60min", 1, 0)
df1$greaterthan60 <- ifelse(df1$MIN_ToBreak == "greaterthan60", 1, 0)

df1$is_white <- ifelse(df1$race_str == "is_white", 1, 0)
df1$is_black <- ifelse(df1$race_str == "is_black", 1, 0)
df1$is_hispanic <- ifelse(df1$race_str == "is_hispanic", 1, 0)
df1$is_asian <- ifelse(df1$race_str == "is_asian", 1, 0)

# list(df1)
# cols <- c("is15min", "is30min", "is60min", "greaterthan60", "is_black", "is_hispanic", "is_asian", "is_white")
# df1[cols] <- lapply(df1[cols], function(x) ifelse(x == "True", 1, 0))

df1 %>% 
  # filter(hospice_use == 1) %>% 
  ggplot(aes(x = is_black)) +
  geom_histogram() +
  facet_wrap(~hospice_use)




# df1 <- df1 %>% 
  # mutate(greaterthan60 = if_else(MIN_ToBreak == "greaterthan60", 1, 0))

df1 <- df1 %>% 
  mutate(Rural = if_else(ruca_category == "Rural", 1, 0))

df1$is60plus <- ifelse(df1$is60min == 1 | df1$greaterthan60 == 1, 1, 0)

df_filtered <- df1 %>% 
  add_count(ZIPCODE) %>% 
  filter(n >= 11) %>% 
  select(-n)




lapply(df_filtered['MIN_ToBreak'], function(x) table(x))



lapply(df_white['greaterthan60'], function(x) table(x))

results <- list()
all_me_df <- list()

model <- glmer(hospice_use ~ is_female + agedgrp5 + is_black + is_hispanic + is_asian + is15min + is30min + 
                 # Black_pct + White_pct + Hispanic_pct +  Asian_pct +
                 Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
               family = binomial,
               data = df_filtered,
               control = glmerControl(optimizer = "bobyqa",
                                      optCtrl = list(maxfun = 1e5)))
               
sink("filtered_zip_model.txt")

summary(model)



sink()

results[['df_filtered']] <- model


marg_eff <- margins(model)
me_df <- summary(marg_eff) %>% 
  rename(Variable = factor, Effect = AME, SE = SE, p = p) %>% 
  mutate(
    lower = Effect - 1.96*SE,
    upper = Effect + 1.96*SE,
    significant = ifelse(p < 0.05, TRUE, FALSE),
    subgroup = "df_filtered"
  )

all_me_df[["df_filtered"]] <- me_df




df_white <- subset(df_filtered, RTI_RACE == 1)
df_black <- subset(df_filtered, RTI_RACE == 2)
df_asian <- subset(df_filtered, RTI_RACE == 4)
df_hispanic <- subset(df_filtered, RTI_RACE == 5)

df_15 <- subset(df_filtered, is15min == TRUE)
df_30 <- subset(df_filtered, is30min == TRUE)
df_60plus <- subset(df_filtered, is60plus == TRUE)



south <- c('AL', 'FL', 'GA', 'KY', 'MI', 'SC', 'NC', 'TN')
south_df <-  df_filtered[df_filtered$STATE_x %in% south, ]


midwest <-  c('IL', 'IN', 'MI', 'MN', 'OH', 'WI')
midwest_df <- df_filtered[df_filtered$STATE_x %in% midwest, ]

southcentral <- c('AR', 'LA', 'NM', 'OK', 'TX')
southcentral_df <- df_filtered[df_filtered$STATE_x %in% southcentral, ]

# received a ConvergenceWarning
central <- c('IA', 'KS', 'MO', 'NE')
central_df <- df_filtered[df_filtered$STATE_x %in% central, ]

# received a ConvergenceWarning
mountain <- c('CO', 'MT', 'ND', 'SD', 'UT', 'WY')
mountain_df <- df_filtered[df_filtered$STATE_x %in% mountain, ]

pacific <- c('AZ', 'CA', 'HI', 'NV')
pacific_df <- df_filtered[df_filtered$STATE_x %in% pacific, ]

pnw <- c('AK', 'ID', 'OR', 'WA')
pnw_df <- df_filtered[df_filtered$STATE_x %in% pnw, ]

newengland <- c('ME', 'NH', 'VT', 'MA', 'RI', 'CT')
newengland_df <- df_filtered[df_filtered$STATE_x %in% newengland, ]

midatlantic <- c('VA', 'MD', 'PA', 'DE', 'WV', 'DC')
midatlantic_df <- df_filtered[df_filtered$STATE_x %in% midatlantic, ]

nynj <- c('NY', 'NJ')
nynj_df <- df_filtered[df_filtered$STATE_x %in% nynj, ]

dfs <- list(df_white = df_white, df_black = df_black, df_asian = df_asian, df_hispanic = df_hispanic)

dfs1 <- list(df_15 = df_15, df_30 = df_30, df_60plus = df_60plus)

dfs2 <- list(south_df = south_df, midwest_df = midwest_df, southcentral_df = southcentral_df, central_df = central_df,
             mountain_df = mountain_df, pacific_df = pacific_df, pnw_df = pnw_df,
             newengland_df = newengland_df, midatlantic_df = midatlantic_df, nynj_df = nynj_df)





for (name in names(dfs)) {
  df <- dfs[[name]]
  model <- glmer(hospice_use ~ is_female + agedgrp5 + is15min + is30min + Black_pct + White_pct + Hispanic_pct +  Asian_pct + Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
                 family = binomial,
                 data = df,
                 control = glmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 1e5)))
  
  
  results[[name]] <- model
  sink(paste0(name, "_model_summary_context.txt"))
  print(summary(model))
  sink()
  
  marg_eff <- margins(model)
  me_df <- summary(marg_eff) %>% 
    rename(Variable = factor, Effect = AME, SE = SE, p = p) %>% 
    mutate(
      lower = Effect - 1.96*SE,
      upper = Effect + 1.96*SE,
      significant = ifelse(p < 0.05, TRUE, FALSE),
      subgroup = name
    )
  all_me_df[[name]] <- me_df
}

for (name in names(dfs1)) {
  df <- dfs1[[name]]
  model <- glmer(hospice_use ~ is_female + agedgrp5 + is_black + is_asian + is_hispanic + Black_pct + White_pct + Hispanic_pct +  Asian_pct+ Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
                 family = binomial,
                 data = df,
                 control = glmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 1e5)))
  
  
  results[[name]] <- model
  sink(paste0(name, "_model_summary_context.txt"))
  print(summary(model))
  sink()
  marg_eff <- margins(model)
  me_df <- summary(marg_eff) %>% 
    rename(Variable = factor, Effect = AME, SE = SE, p = p) %>% 
    mutate(
      lower = Effect - 1.96*SE,
      upper = Effect + 1.96*SE,
      significant = ifelse(p < 0.05, TRUE, FALSE),
      subgroup = name
    )
  all_me_df[[name]] <- me_df
}

for (name in names(dfs2)) {
  df <- dfs2[[name]]
  model <- glmer(hospice_use ~ is_female + agedgrp5 + is_black + is_asian + is_hispanic + is15min + is30min + Black_pct + White_pct + Hispanic_pct +  Asian_pct+ Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
                 family = binomial,
                 data = df,
                 control = glmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 1e5)))
  
  
  results[[name]] <- model
  sink(paste0(name, "_model_summary_context.txt"))
  print(summary(model))
  sink()
  marg_eff <- margins(model)
  me_df <- summary(marg_eff) %>% 
    rename(Variable = factor, Effect = AME, SE = SE, p = p) %>% 
    mutate(
      lower = Effect - 1.96*SE,
      upper = Effect + 1.96*SE,
      significant = ifelse(p < 0.05, TRUE, FALSE),
      subgroup = name
    )
  all_me_df[[name]] <- me_df
}

all_me_df_combined <- bind_rows(all_me_df)
write.csv(all_me_df_combined, "marg_effects_all_subgroups.csv", row.names = FALSE)


results['south_df']

names <- c('df_filtered', dfs, dfs1, dfs2)



tidy_results <- tidy(modest, effects = "fixed", conf.int=TRUE)
print(tidy_results)


tidy_all <- purrr::map2_df(
  results,
  names(names),
  ~ tidy(.x, effects = "fixed", conf.int = TRUE) %>% 
    mutate(model = .y)
  
)

tidy_all <- tidy_all %>% 
  mutate(
    significant = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ "ns"
    )) %>% 
  select(model, everything())

  

write.csv(tidy_all, "submodel_results.csv", row.names = FALSE)


names[["df_filtered"]] <- df_filtered
# 
# var_names <- c("SEX","agedgrp5", "RTI_RACE","hospice_use",
#                "ruca_category", "MIN_ToBreak", "Region", "race_str",'is_white',"is_asian",                     
#                "is_black", "is_hispanic", "Suburban", "Urban","is_female", "is15min", "is30min", "greaterthan60", 'is60plus')                       
# 
# dfs2 <- list(south_df = south_df, midwest_df = midwest_df, southcentral_df = southcentral_df, central_df = central_df,
#              mountain_df = mountain_df, pacific_df = pacific_df, pnw_df = pnw_df,
#              newengland_df = newengland_df, midatlantic_df = midatlantic_df, nynj_df = nynj_df, df_filtered = df_filtered)



var_names <- c("SEX","agedgrp5", "RTI_RACE","hospice_use",
               "ruca_category", "MIN_ToBreak", "Region", "race_str",'is_white',"is_asian",                     
               "is_black", "is_hispanic", "Suburban", "Urban","is_female", "is15min", "is30min", "greaterthan60", 'is60plus')                       

dfs2 <- list(df_filtered = df_filtered, df_white = df_white, df_black = df_black, df_asian = df_asian, 
             df_hispanic = df_hispanic)

sink("data_summary1.txt")
for (name in names(dfs2)) {
  df <- dfs2[[name]][var_names]
  
  cat("\n==========================\n")
  cat("Dataset:", name, "\n")
  cat("Shape:", nrow(df), "rows x", ncol(df), "columns\n\n")
  for (col in names(df)) {
    cat("Variable:", col, "\n")
    print(table(df[[col]], useNA = "ifany"))
    cat("\n")
  }
}
sink()

midatlantic_df


colnames(df_filtered)



# Interaction Analysis ----------------------------------------------------

sink("interaction_analysis.txt")

df_filtered$race_str <- as.factor(df_filtered$race_str)

levels(df_filtered$srace_str)

df_filtered$race_str <- relevel(df_filtered$race_str, ref="is_white")


df_filtered$dist_factor <- NA


df_filtered$dist_factor[df_filtered$is15min == 1] <- '15'
df_filtered$dist_factor[df_filtered$is30min == 1] <- '30'
df_filtered$dist_factor[df_filtered$is60min == 1] <- '60'

df_filtered$dist_factor <- factor(df_filtered$dist_factor, levels = c("15", '30', '60'))


df_filtered$dist_factor <- relevel(df_filtered$dist_factor, ref="60")


# Race --------------------------------------------------------------------



ModIntRace <- glmer(
  hospice_use ~ race_str * (is_female + 
                              agedgrp5 + 
                              dist_factor
                            + Black_pct + White_pct + Hispanic_pct +  Asian_pct + 
                              Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
  data = df_filtered,
  family = binomial
)

ModIntRace1 <- glmer(
  hospice_use ~ race_str * (is_female + agedgrp5 + 
                              dist_factor + 
                              # Black_pct + White_pct + Hispanic_pct +  Asian_pct + 
                              Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
  data = df_filtered,
  family = binomial
)



# is_female + agedgrp5 + is15min + is30min + Black_pct + White_pct + Hispanic_pct +  Asian_pct + Suburban + Urban + Income_log_scaled + (1|ZIPCODE)

print("ModIntRace - summary")

summary(ModIntRace)

print("ModIntRace1 - summary")
summary(ModIntRace1)





# Distance ----------------------------------------------------------------




ModIntDist <- glmer(hospice_use ~ dist_factor * (is_female + agedgrp5 + race_str
                                                 + Black_pct + White_pct + Hispanic_pct +  Asian_pct+ 
                                                   Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
               family = binomial,
               data = df_filtered,
               control = glmerControl(optimizer = "bobyqa",
                                      optCtrl = list(maxfun = 1e5)))

ModIntDist1 <- glmer(hospice_use ~ dist_factor * (is_female + agedgrp5 + race_str +
                                                 # + Black_pct + White_pct + Hispanic_pct +  Asian_pct+ 
                                                   Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
                    family = binomial,
                    data = df_filtered,
                    control = glmerControl(optimizer = "bobyqa",
                                           optCtrl = list(maxfun = 1e5)))

# is_female + agedgrp5 + is15min + is30min + Black_pct + White_pct + Hispanic_pct +  Asian_pct + Suburban + Urban + Income_log_scaled + (1|ZIPCODE)

print("ModIntDist - summary")
summary(ModIntDist)


print("ModIntDist1 - summary")
summary(ModIntDist1)


# Region ------------------------------------------------------------------


df_filtered <- df_filtered %>% 
  mutate(region = case_when(
    STATE_x %in% midatlantic ~ "midatlantic",
    STATE_x %in% midwest ~ "midwest",
    STATE_x %in% mountain ~ "mountain",
    STATE_x %in% newengland ~ "newengland",
    STATE_x %in% nynj ~ "nynj",
    STATE_x %in% pacific ~ "pacific",
    STATE_x %in% pnw ~ "pnw",
    STATE_x %in% south ~ "south",
    STATE_x %in% southcentral ~ "southcentral",
    TRUE ~ NA_character_
  )) %>% 
  mutate(region = factor(region))


ModIntReg <- glmer(hospice_use ~ region * (is_female + agedgrp5 + 
                                             race_str + 
                                             dist_factor +
                                             Black_pct + White_pct + Hispanic_pct +  Asian_pct+ 
                                             Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
               family = binomial,
               data = df_filtered,
               control = glmerControl(optimizer = "bobyqa",
                                      optCtrl = list(maxfun = 1e5)))

ModIntReg1 <- glmer(hospice_use ~ region * (is_female + agedgrp5 + 
                                             race_str + 
                                             dist_factor +
                                             # Black_pct + White_pct + Hispanic_pct +  Asian_pct+ 
                                             Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
                   family = binomial,
                   data = df_filtered,
                   control = glmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 1e5)))


# Main --------------------------------------------------------------------



ModMain <- glmer(hospice_use ~ is_female + agedgrp5 + 
                                                      race_str + 
                                                      dist_factor +
                                                      Black_pct + White_pct + Hispanic_pct +  Asian_pct+
                                                      Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
                            family = binomial,
                            data = df_filtered,
                            control = glmerControl(optimizer = "bobyqa",
                                                   optCtrl = list(maxfun = 1e5)))


ModMain1 <- glmer(hospice_use ~ is_female + agedgrp5 + 
                   race_str + 
                   dist_factor +
                   # Black_pct + White_pct + Hispanic_pct +  Asian_pct+
                   Suburban + Urban + Income_log_scaled + (1|ZIPCODE),
family = binomial,
data = df_filtered,
control = glmerControl(optimizer = "bobyqa",
                       optCtrl = list(maxfun = 1e5)))


ModMainReg <- glmer(hospice_use ~ is_female + agedgrp5 + 
                   race_str + 
                   dist_factor +
                   Black_pct + White_pct + Hispanic_pct +  Asian_pct+
                   Suburban + Urban + Income_log_scaled + region + (1|ZIPCODE),
family = binomial,
data = df_filtered,
control = glmerControl(optimizer = "bobyqa",
                       optCtrl = list(maxfun = 1e5)))



ModMainReg1 <- glmer(hospice_use ~ is_female + agedgrp5 + 
                      race_str + 
                      dist_factor +
                      # Black_pct + White_pct + Hispanic_pct +  Asian_pct+
                      Suburban + Urban + Income_log_scaled + region + (1|ZIPCODE),
                    family = binomial,
                    data = df_filtered,
                    control = glmerControl(optimizer = "bobyqa",
                                           optCtrl = list(maxfun = 1e5)))

print("ModMain - summary")
summary(ModMain)

print("ModMain1 - summary")
summary(ModMain1)

# Compare -----------------------------------------------------------------


print("ModMain/Race - ANOVA")
stats::anova(ModMain, ModIntRace)

print("ModMain1/Race1 - ANOVA")
stats::anova(ModMain1, ModIntRace1)

print("ModMain/Dist - ANOVA")
stats::anova(ModMain, ModIntDist)

print("ModMain1/Dist1 - ANOVA")
stats::anova(ModMain1, ModIntDist1)

stats::anova(ModMainReg, ModIntReg)
stats::anova(ModMainReg1, ModIntReg1)


print("ModIntRace - ANOVA")
Anova(ModIntRace, type = "III", test.statistic = "Chisq")

print("ModIntRace1 - ANOVA")
Anova(ModIntRace1, type = "III", test.statistic = "Chisq")

print("ModIntDist - ANOVA")
Anova(ModIntDist, type = "III", test.statistic = "Chisq")

print("ModIntDist1 - ANOVA")
Anova(ModIntDist1, type = "III", test.statistic = "Chisq")

print("ModMain - ANOVA")
Anova(ModMain, type = "III", test.statistic = "Chisq")

print("ModMain1 - ANOVA")
Anova(ModMain1, type = "III", test.statistic = "Chisq")



sink()


X <- model.matrix(~region * (is_female + agedgrp5 + 
                               race_str + 
                               dist_factor +
                               Black_pct + White_pct + Hispanic_pct +  Asian_pct+ 
                               Suburban + Urban + Income_log_scaled) + (1|ZIPCODE),
                  family = binomial,
                  data = df_filtered,
                  control = glmerControl(optimizer = "bobyqa",
                                         optCtrl = list(maxfun = 1e5)))

