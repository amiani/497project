relu <- function(input) {
  return(pmax(input, 0))
}

act <- function(params, ob) { #forward pass
  h <- unlist(ob)
  for (layer in params[1:length(params)-1]) {
    h <- relu(layer%*%h)
  }
  out <- params[[length(params)]] %*% h
  return(which.max(out)-1)
}

makePopulation <- function(parent, sdev, popSize) {
  N <- length(parent)
  noise <- matrix(rnorm(N*popSize/2), N, popSize/2)
  noise <- cbind(noise, -noise) #mirrored sampling
  population <- parent + sdev * noise
  return(list(population, noise))
}

initParams <- function(layerWidths) {
  numParams <- 0
  prev <- layerWidths[[1]]
  for (w in layerWidths[-1]) {
    numParams <- numParams + w * prev
    prev <- w
  }
  parent <- rnorm(numParams)
  return(parent)
}

updateParent <- function(parent, sdev, fitness, noise, alpha) {
  print(fitness)
  parent <- parent + alpha * apply(fitness * noise, 1, mean) / sdev
  return(parent)
}

vecToLayers <- function(params, layerWidths) {
  layers <- list()
  prev <- layerWidths[[1]]
  for (i in 2:length(layerWidths)) {
    w <- layerWidths[[i]]
    n <- prev * w
    layers[[i-1]] <- matrix(params[1:n], w, prev)
    prev <- w
    params <- params[-n]
  }
  return(layers)
}

library(gym)

setupGym <- function(env_id, isMonitor) {
  remote_base <- "http://127.0.0.1:5000"
  client <- create_GymClient(remote_base)
  #print(client)

  # Create environment
  instance_id <- env_create(client, env_id)
  #print(instance_id)

  # List all environments
  all_envs <- env_list_all(client)
  #print(all_envs)

  # Set up agent
  action_space_info <- env_action_space_info(client, instance_id)
  observation_space_info <- env_observation_space_info(client, instance_id)

  layerWidths <- list(
    action_space_info[["n"]],
    64,
    64,
    observation_space_info[["n"]])

  # Run experiment, with monitor
  outdir <- "/tmp/random-agent-results"
  if (isMonitor) {
    env_monitor_start(client, instance_id, outdir, force = TRUE, resume = FALSE)
  }
  return(list(client, instance_id, observation_space_info[["shape"]], action_space_info[["n"]]))
}

evaluateAgent <- function(agent, client, instance_id) {
  episode_count <- 1
  max_steps <- 200
  reward <- 0
  done <- FALSE

  for (i in 1:episode_count) {
    ob <- env_reset(client, instance_id)
    for (i in 1:max_steps) {
      action <- act(agent, ob)
      results <- env_step(client, instance_id, action, render = FALSE)
      reward <- reward + results[["reward"]]
      ob <- results[["observation"]]
      if (results[["done"]]) break
    }
  }
  return(reward)
}

evaluatePopulation <- function(population, client, instance_id) {
  fitnesses <- lapply(population, evaluateAgent, client, instance_id)
  return(fitnesses)
}

envData <- setupGym("CartPole-v1", FALSE)
client <- envData[[1]]
instance_id <- envData[[2]]
hiddenDim <- 64
sdev <- 1
alpha <- 0.01
layerWidths <- list(
  unlist(envData[[3]]),
  hiddenDim,
  hiddenDim,
  envData[[4]])
parent <- initParams(layerWidths)

parentFitness <- 0
while(parentFitness < 150) {
  popData <- makePopulation(parent, sdev, 10)
  popAgents <- apply(popData[[1]], 2, vecToLayers, layerWidths)
  fitness <- evaluatePopulation(popAgents, client, instance_id)
  parent <- updateParent(parent, sdev, unlist(fitness), popData[[2]], alpha)
  parentFitness <- evaluateAgent(parent, client, instance_id)
  print(parentFitness)
}

# Dump result info to disk
env_monitor_close(client, instance_id)