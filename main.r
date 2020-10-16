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

makePopulation <- function(meanVec, sdev, popSize) {
  N <- length(meanVec)
  noise <- matrix(rnorm(N*popSize/2), N, popSize/2)
  noise <- cbind(noise, -noise) #mirrored sampling
  population <- meanVec + sdev * noise
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

updateParent <- function(parent, sdev, fitnesses, noise, alpha) {
  parent <- parent + alpha * apply(fitnesses * noise, 1, mean) / sdev
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
  print(action_space_info)

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
      results <- env_step(client, instance_id, action, render = TRUE)
      reward <- reward + results[["reward"]]
      ob <- results[["observation"]]
      if (results[["done"]]) break
    }
  }
  return(reward)
}

envData <- setupGym("CartPole-v1", TRUE)
layerWidths <- list(
  unlist(envData[[3]]),
  64,
  64,
  envData[[4]])
parent <- initParams(layerWidths)
parentAgent <- vecToLayers(parent, layerWidths)
fitness <- evaluateAgent(parentAgent, envData[[1]], envData[[2]])
print(fitness)

# Dump result info to disk
#env_monitor_close(client, instance_id)