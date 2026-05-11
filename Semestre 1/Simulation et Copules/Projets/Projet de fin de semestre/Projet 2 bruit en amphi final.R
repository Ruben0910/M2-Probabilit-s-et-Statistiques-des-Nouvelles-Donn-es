###############################################################
###                                                         ###
###       Projet 2 - Simulations et Copules                 ###
###                                                         ###
###             Bruit en Amphi                              ###
###                                                         ###
###############################################################

### Partie 1 ###
#Dans cette partie on considere un nombre fixe d'eleve par groupe 


#nombre de place dans l'amphitheatre
N = 90

#nombre de groupe d'élèves
L = 4

#Matrice du bruit en fonction du voisinage
groupes <- matrix(0, nrow=4, ncol=4)
valeurs <- c("A", "B", "C", "D")
rownames(groupes) <- colnames(groupes) <- valeurs

# A avec ...
groupes["A", "A"] <- 10
groupes["A", "B"] <- 8;  groupes["B", "A"] <- 8
groupes["A", "C"] <- 5;  groupes["C", "A"] <- 5
groupes["A", "D"] <- 0;  groupes["D", "A"] <- 0

# B avec ...
groupes["B", "B"] <- 6
groupes["B", "C"] <- 3;  groupes["C", "B"] <- 3
groupes["B", "D"] <- 0;  groupes["D", "B"] <- 0

# C avec ...
groupes["C", "C"] <- 2
groupes["C", "D"] <- 0;  groupes["D", "C"] <- 0

# D avec D (reste 0)
groupes["D", "D"] <- 0



#matrice amphitheatre 
rangs = 9
col = 10

A = 40
B=16
C= 24
D = 10

bruit_voisin_carre = function(coord,G){
  P = arrayInd(coord, .dim = dim(G))
  r = P[1]
  c = P[2]
  b = 0
  vois = c()
  for(i in -1:1){
    for (j in -1:1){
      #on enleve la coordonée elle meme 
      if (i==0 && j == 0) next
      vr = r + i
      vc = c + j
      if (vr>=1 && vr<= nrow(G) && vc>= 1 && vc <= ncol(G)){
        b = b + groupes[G[r,c],G[vr,vc]]
        vois = c(vois, (vc - 1) * nrow(G) + vr)
      }
    }
  }
  return(list(V=vois,B=b))
}

H_carre = function(G){
  N = length(G)
  H = 0
  for (i in 1:N){
    Vi = bruit_voisin_carre(i,G)
    for (v in Vi$V){
      if (v > i){ H = H + groupes[G[i], G[v]] }
    }
  }
  return(H)
}


transition_carre = function(G,b){
  N= length(G)
  G_new = G
  coord = sample(1:N,2,replace = FALSE)
  G_new[coord[1]] = G[coord[2]]
  G_new[coord[2]] = G[coord[1]]
  u = runif(1)
  b_old1 = bruit_voisin_carre(coord[1],G)
  b_old2 = bruit_voisin_carre(coord[2],G)
  b_new1 = bruit_voisin_carre(coord[1],G_new)
  b_new2 = bruit_voisin_carre(coord[2],G_new)
  d = ( (b_new1$B+b_new2$B) - (b_old1$B+b_old2$B))
  if (coord[1] %in% (b_old2$V)) {
    r = groupes[G[coord[1]],G[coord[2]]]
    d = ( (b_new1$B+b_new2$B - r) - (b_old1$B+b_old2$B - r))
  }
  alpha = exp(-b*d)
  if (u>alpha) {
    G_new = G
    d = 0
  }
  return(list(G=G_new,D = d))
}


RS_carre = function(n,B0){
  vecteur_donnees <- c(rep("A", A), 
                       rep("B", B), 
                       rep("C", C), 
                       rep("D", D))
  
  
  G <- matrix(vecteur_donnees, nrow = rangs, ncol = col)
  H_evol = rep(0,n)
  H_evol[1] = H_carre(G)
  H_best = H_evol[1]
  G_best = G
  for (i in 2:n){
    B = B0*log(i)
    toto = transition_carre(G,B)
    G = toto$G
    H_evol[i] = H_evol[i-1]+toto$D
    if (H_evol[i]<H_best){
      H_best = H_evol[i]
      G_best = G
    }
  }
  
  return(list(G=G_best,H=H_best,E=H_evol ))
}

toto = RS_carre(10000,0.2)
plot(toto$E,type='l')
print(toto$H)
print(H_carre(toto$G))


visualiser_amphi = function(G,groupes,H_best){
  matrice_bruit = matrix(0, nrow = rangs, ncol = col)
  for (i in (1:N)){
    bruit = bruit_voisin_carre(i,G)
    matrice_bruit[i] = bruit$B
  }
  df_bruit = melt(matrice_bruit)
  colnames(df_bruit)=c("Rang","Place","Niveau_de_Bruit")
  df_type = melt(G)
  colnames(df_type)=c("Rang","Place","Type")
  df_final = data.frame(df_type,Bruit = df_bruit$Niveau_de_Bruit)
  
  amphi =  ggplot(df_final, aes(x = Place, y = Rang, fill = Bruit)) +
    geom_tile(color = "white", lwd = 1) + 
    geom_text(aes(label = Type), color = "black", fontface = "bold") + 
    scale_fill_gradient2(low = "#66CCFF", mid = "#FFCC00", high = "#FF3333", 
                         midpoint = max(df_final$Bruit)/2, name = "Niveau de Bruit") +
    theme_minimal() +
    labs(title = "Carte du Bruit Optimisée dans l'Amphithéâtre",
         subtitle = paste("Bruit Total =", H_best), 
         x = "Numéro de Place", y = "Rangée") +
    theme(panel.grid = element_blank(),
          axis.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, face="bold"))
  
  print(amphi)
}
library(reshape2)
library(tidyverse)

visualiser_amphi_types = function(G, H_best){
  df_type = melt(G)
  colnames(df_type) = c("Rang", "Place", "Type")

  amphi_types = ggplot(df_type, aes(x = Place, y = Rang, fill = Type)) +
    geom_tile(color = "white", lwd = 1) + 
    geom_text(aes(label = Type), color = "black", fontface = "bold") + 
    scale_fill_manual(values = c("A" = "#FF9999",  
                                 "B" = "#FFCC99",  
                                 "C" = "#99CC99",  
                                 "D" = "#99CCFF")) +
    
    theme_minimal() +
    labs(title = "Répartition des Groupes dans l'Amphithéâtre",
         subtitle = paste("Configuration actuelle - Score de Bruit Total =", H_best), 
         x = "Numéro de Place", y = "Rangée",
         fill = "Groupe") +
    theme(panel.grid = element_blank(),
          axis.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, face="bold"))
  
  print(amphi_types)
}

par(mfrow = c(1,2))
visualiser_amphi(toto$G,groupes,toto$H)
visualiser_amphi_types(toto$G, toto$H)











## Partie 2 : On laisse à l'utilisateur de définir lui-même ses variables via une application Rshiny

library(shiny)

# 1. DÉFINITION DE LA MATRICE DE RÉFÉRENCE (Invisible dans l'UI)
groupes_ref <- matrix(0, nrow=4, ncol=4)
valeurs <- c("A", "B", "C", "D")
rownames(groupes_ref) <- colnames(groupes_ref) <- valeurs

groupes_ref["A", "A"] <- 10; groupes_ref["A", "B"] <- 8; groupes_ref["B", "A"] <- 8
groupes_ref["A", "C"] <- 5;  groupes_ref["C", "A"] <- 5; groupes_ref["A", "D"] <- 0
groupes_ref["D", "A"] <- 0;  groupes_ref["B", "B"] <- 6; groupes_ref["B", "C"] <- 3
groupes_ref["C", "B"] <- 3;  groupes_ref["B", "D"] <- 0; groupes_ref["D", "B"] <- 0
groupes_ref["C", "C"] <- 2;  groupes_ref["C", "D"] <- 0; groupes_ref["D", "C"] <- 0
groupes_ref["D", "D"] <- 0

# 2. LOGIQUE MÉTIER (Recuit Simulé)
bruit_voisin_carre = function(coord, G, matrice_bruit_ref){
  P = arrayInd(coord, .dim = dim(G))
  r = P[1]
  c = P[2]
  b = 0
  vois = c()
  
  for(i in -1:1){
    for (j in -1:1){
      if (i==0 && j == 0) next
      vr = r + i
      vc = c + j
      if (vr>=1 && vr<= nrow(G) && vc>= 1 && vc <= ncol(G)){
        b = b + matrice_bruit_ref[G[r,c], G[vr,vc]]
        vois = c(vois, (vc - 1) * nrow(G) + vr)
      }
    }
  }
  return(list(V=vois, B=b))
}

H_carre = function(G, matrice_bruit_ref){
  N_len = length(G)
  H = 0
  for (i in 1:N_len){
    Vi = bruit_voisin_carre(i, G, matrice_bruit_ref)
    for (v in Vi$V){
      if (v > i){ H = H + matrice_bruit_ref[G[i], G[v]] }
    }
  }
  return(H)
}

transition_carre = function(G, b, matrice_bruit_ref){
  N_len = length(G)
  G_new = G
  coord = sample(1:N_len, 2, replace = FALSE)
  
  G_new[coord[1]] = G[coord[2]]
  G_new[coord[2]] = G[coord[1]]
  
  u = runif(1)
  
  b_old1 = bruit_voisin_carre(coord[1], G, matrice_bruit_ref)
  b_old2 = bruit_voisin_carre(coord[2], G, matrice_bruit_ref)
  b_new1 = bruit_voisin_carre(coord[1], G_new, matrice_bruit_ref)
  b_new2 = bruit_voisin_carre(coord[2], G_new, matrice_bruit_ref)
  
  delta = ((b_new1$B + b_new2$B) - (b_old1$B + b_old2$B))
  
  if (coord[1] %in% (b_old2$V)) {
    r_val = matrice_bruit_ref[G[coord[1]], G[coord[2]]]
    delta = ((b_new1$B + b_new2$B - r_val) - (b_old1$B + b_old2$B - r_val))
  }
  
  alpha = exp(-b * delta)
  
  if (u > alpha) {
    G_new = G
    delta = 0
  }
  
  return(list(G = G_new, D = delta))
}

# 3. UI (INTERFACE UTILISATEUR)
ui <- fluidPage(
  titlePanel("Optimisation de placement en Amphithéâtre"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Paramètres de l'Amphithéâtre"),
      numericInput("rows", "Nombre de Rangs :", value = 9, min = 2),
      numericInput("cols", "Nombre de Places par Rang :", value = 10, min = 2),
      
      hr(),
      h4("Répartition des Élèves"),
      fluidRow(
        column(6, numericInput("nA", "Groupe A :", value = 40)),
        column(6, numericInput("nB", "Groupe B :", value = 16))
      ),
      fluidRow(
        column(6, numericInput("nC", "Groupe C :", value = 24)),
        column(6, numericInput("nD", "Groupe D :", value = 10))
      ),
      uiOutput("validation_msg"),
      
      hr(),
      h4("Simulation"),
      numericInput("iter", "Itérations :", value = 5000, step = 1000),
      numericInput("beta0", "Refroidissement (B0) :", value = 0.2, step = 0.1),
      
      hr(),
      actionButton("run", "Lancer l'Optimisation", class = "btn-primary", width = "100%")
    ),
    
    mainPanel(
      # On a retiré l'onglet "Matrice" pour ne garder que la visualisation
      tabsetPanel(
        tabPanel("Résultats Visuels",
                 br(),
                 fluidRow(
                   column(6, plotOutput("plot_types", height = "400px")),
                   column(6, plotOutput("plot_noise", height = "400px"))
                 ),
                 hr(),
                 h4("Évolution de l'énergie"),
                 plotOutput("plot_evol", height = "300px")
        )
      )
    )
  )
)


# 4. SERVER
server <- function(input, output, session) {
  output$validation_msg <- renderUI({
    total_places <- input$rows * input$cols
    total_eleves <- input$nA + input$nB + input$nC + input$nD
    
    if (total_places == total_eleves) {
      tags$p(icon("check"), " La somme correspond (", total_places, ")", style = "color: green;")
    } else {
      tags$p(icon("exclamation-triangle"), paste("Élèves:", total_eleves, "/ Places:", total_places), style = "color: red; font-weight: bold;")
    }
  })
  
  results <- reactiveValues(G = NULL, H = NULL, E = NULL, ran = FALSE)
  
  observeEvent(input$run, {
    total_places <- input$rows * input$cols
    total_eleves <- input$nA + input$nB + input$nC + input$nD
    
    if (total_places != total_eleves) {
      showNotification("Erreur : Le nombre d'élèves ne correspond pas aux places !", type = "error")
      return()
    }
    
    withProgress(message = 'Optimisation...', value = 0, {
      
      vecteur_donnees <- c(rep("A", input$nA), rep("B", input$nB), 
                           rep("C", input$nC), rep("D", input$nD))
      
      vecteur_donnees <- sample(vecteur_donnees) 
      G <- matrix(vecteur_donnees, nrow = input$rows, ncol = input$cols)
      
      n_iter <- input$iter
      B0_val <- input$beta0
      
      H_evol <- numeric(n_iter)
      H_evol[1] <- H_carre(G, groupes_ref)
      
      H_best <- H_evol[1]
      G_best <- G
      
      update_step <- max(1, n_iter / 10)
      
      for (i in 2:n_iter){
        B <- B0_val * log(i)
        res <- transition_carre(G, B, groupes_ref)
        G <- res$G
        H_evol[i] <- H_evol[i-1] + res$D
        
        if (H_evol[i] < H_best){
          H_best <- H_evol[i]
          G_best <- G
        }
        if (i %% update_step == 0) incProgress(1/10)
      }
      
      results$G <- G_best
      results$H <- H_best
      results$E <- H_evol
      results$ran <- TRUE
    })
  })
  
  # --- GRAPHIQUE 1 : REPARTITION (TYPES) ---
  output$plot_types <- renderPlot({
    req(results$ran)
    G <- results$G
    df_type <- melt(G)
    colnames(df_type) <- c("Rang", "Place", "Type")
    
    ggplot(df_type, aes(x = Place, y = Rang, fill = Type)) +
      geom_tile(color = "white", lwd = 1) + 
      geom_text(aes(label = Type), color = "black", fontface = "bold") + 
      scale_fill_manual(values = c("A"="#FF9999", "B"="#FFCC99", "C"="#99CC99", "D"="#99CCFF")) +
      scale_y_reverse() +
      theme_minimal() +
      labs(title = "Répartition Optimisée", subtitle = paste("Score :", results$H)) +
      theme(panel.grid = element_blank())
  })
  
  # --- GRAPHIQUE 2 : CARTE DU BRUIT ---
  output$plot_noise <- renderPlot({
    req(results$ran)
    G <- results$G

    mat_visu <- matrix(0, nrow=nrow(G), ncol=ncol(G))
    for (r in 1:nrow(G)){
      for (c in 1:ncol(G)){
        idx <- (c - 1) * nrow(G) + r
        bruit <- bruit_voisin_carre(idx, G, groupes_ref)
        mat_visu[r,c] <- bruit$B
      }
    }
    
    df_bruit <- melt(mat_visu)
    colnames(df_bruit) <- c("Rang", "Place", "Niveau")
    
    df_types <- melt(G)
    colnames(df_types) <- c("Rang", "Place", "Type")
    
    df_final <- data.frame(df_bruit, Type = df_types$Type)
    
    ggplot(df_final, aes(x = Place, y = Rang, fill = Niveau)) +
      geom_tile(color = "white", lwd = 1) +
      geom_text(aes(label = Type), color = "black", fontface = "bold") +
      scale_fill_gradient2(low = "#66CCFF", mid = "#FFCC00", high = "#FF3333", 
                           midpoint = max(df_final$Niveau)/2) +
      scale_y_reverse() +
      theme_minimal() +
      labs(title = "Carte de Chaleur (Bruit)") +
      theme(panel.grid = element_blank())
  })
  
  output$plot_evol <- renderPlot({
    req(results$ran)
    plot(results$E, type='l', col='blue', lwd=1, main="Convergence", ylab="Énergie", xlab="Itérations")
    grid()
  })
}

shinyApp(ui, server)