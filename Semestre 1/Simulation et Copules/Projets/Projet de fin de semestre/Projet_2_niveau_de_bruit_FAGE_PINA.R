#########################################################
###                                                   ###
###       Projet 2 - Simulations et Copules           ###
###                                                   ###
###             Bruit en Amphi                        ###
###                                                   ###
#########################################################

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
B = 24
C= 16
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

toto = RS_carre(100000,0.07)
plot(toto$E,type='l')
print(toto$H)
print(H_carre(toto$G))
par(mfrow = c(1,1))


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

visualiser_amphi(toto$G,groupes,toto$H)
visualiser_amphi_types(toto$G, toto$H)

