# ===================================================================
# PROJET 1 : SIMULATION ET COPULES
#
# FAGE Annaelle / PINA Ruben
# ===================================================================

set.seed(42)

#parametre
n=1000
N=200
a=-1
b= 1
c=-1
d= 1
e= 0
g= exp(2)
A =(b-a)*(d-c)*(g-e)
teta = 1.95

#simulation
U = matrix(runif(N*n,a,b),ncol=N)
V = matrix(runif(N*n,c,d),ncol=N)
W <- matrix(runif(N * n, e, g), ncol = N)

f=function(x,y) exp(y^2 +x*y)
library(pracma)
I = quad2d(f,a,b,c,d)

p = function(x,y) (0.8*abs(x)+0.1)*(0.8*abs(y)+0.1)

library(rgl)

x = seq(a,b, by = 0.1)
y = seq(c, d, by = 0.1)
z = outer(x,y,f)
z_p = outer(x,y,p)
persp(x,y,z)
#persp3d(x,y,z, col = "blue")
#title3d(main = "Fonction f(x,y) (bleu) et Densite p(x,y) (rouge)")
#surface3d(x, y, z_p, col = "red")

P_inv <- function(Z) {
  val1 <- pmax(0, 0.81 - 1.6 * Z)
  val2 <- pmax(0, -0.81 + 1.6 * Z)
  
  sqrt1 <- sqrt(val1)
  sqrt2 <- sqrt(val2)
  
  toto <- (Z <= 0.5) * ((0.1 - sqrt1) / 0.8) + (Z > 0.5) * ((-0.1 + sqrt2) / 0.8)
  
  return(toto)
}

### Methode 1 ###

#simulation
X1 = (W < f(U,V))

#evolution de l'estimateur
I1_evol  = A*cumsum(X1[,1])/(1:n)

#vecteur
calc_1 = function(ech) A*mean(ech)
vect1 = apply(X1,MARGIN = 2,FUN = calc_1)

### Methode 2 ###

#evolution de l'estimateur
I2_evol = (b-a)*(d-c)*cumsum(f(U[,1],V[,1]))/(1:n)

#vecteur
calc_2 = function(ech1,ech2) (b-a)*(d-c)*mean(f(ech1,ech2))
vect2  = apply(U,V,MARGIN = 2,FUN=calc_2)

### Methode 3 ###
#simulation
Z = matrix(runif(N*n),ncol=N)
S = matrix(runif(N*n),ncol=N)
P_U = P_inv(Z)
P_V = P_inv(S)

#evolution de l'estimateur
I3_evol = cumsum(f(P_U[,1],P_V[,1])/p(P_U[,1],P_V[,1]))/(1:n)

#vecteur
calc3 = function(ech1,ech2) mean(f(ech1,ech2)/p(ech1,ech2))
vect3  = apply(P_U,P_V,MARGIN = 2,FUN=calc3)

### Methode 4 ###~
#simulation du copule de Franck
library(copula)

r_franck = function(n,teta){
  toto= rCopula(n, frankCopula(teta)) 
  return(list(X = qunif(toto[,1],-1,1),Y =qunif(toto[,2],-1,1)))
}

frank_d <- function(u, v, theta) {
  if (theta == 0) stop
  if (any(u < 0 | u > 1) | any(v < 0 | v > 1)) stop
  num <- theta * exp(-theta * (u + v)) * (1 - exp(-theta))
  denom <- ( (1 - exp(-theta * u)) * (1 - exp(-theta * v)) + exp(-theta) - 1 )^2
  
  density <- num / denom
  return(density)
}

Fr = r_franck(N*n,teta)
P_U2 = matrix(Fr$X,ncol=N)
P_V2 = matrix(Fr$Y,ncol=N)

#densite
d_1= function(ech1,ech2){
  A = dunif(ech1,-1,1)
  B = dunif(ech2,-1,1)
  C = punif(ech1,-1,1)
  D = punif(ech2,-1,1)
  return(A*B*frank_d(C,D,teta))

}

#evolution de l'estimateur
I4_evol = cumsum(f(P_U2[,1],P_V2[,1])/d_1(P_U2[,1],P_V2[,1]))/(1:n)

#vecteur
calc4 = function(ech1,ech2) mean(f(ech1,ech2)/d_1(ech1,ech2))
vect4  = apply(P_U2,P_V2,MARGIN = 2,FUN=calc4)

### Methode 5 ###~
#simulation du copule de Franck avec P1 et P2
library(copula)

r_franck_2 = function(n,teta){
  toto= rCopula(n, frankCopula(teta)) 
  return(list(X = P_inv(toto[,1]),Y =P_inv(toto[,2])))
}

F_P = function(x) {
  ifelse(x < -1, 0,
         ifelse(x < 0, -0.4 * x^2 + 0.1 * x + 0.5,
                ifelse(x <= 1, 0.5 + 0.4 * x^2 + 0.1 * x, 1)
         )
  )
}

Fr_p = r_franck_2(N*n,teta)
P_U3 = matrix(Fr_p$X,ncol=N)
P_V3 = matrix(Fr_p$Y,ncol=N)

#densite
d_2= function(ech1,ech2){
  A = p(ech1,ech2)
  C = F_P(ech1)
  D = F_P(ech2)
  return(A*frank_d(C,D,teta))
  
}

#evolution de l'estimateur
I5_evol = cumsum(f(P_U3[,1],P_V3[,1])/d_2(P_U3[,1],P_V3[,1]))/(1:n)

#vecteur
calc5 = function(ech1,ech2) mean(f(ech1,ech2)/d_2(ech1,ech2))
vect5  = apply(P_U3,P_V3,MARGIN = 2,FUN=calc5)


#Sortie graphique
par(mfrow = c(2,3))
plot(I1_evol,type='lines',col='red',ylim=c(I-2,I+2), 
     main = "Evolution des estimateurs", xlab = "n", ylab = "I_chap")
lines(I2_evol,col='blue')
lines(I3_evol,col='green')
lines(I4_evol,col='purple')
lines(I5_evol,col='turquoise')
abline(h=I,col='black')
legend("topright", legend = c("Méthode 1", "Méthode 2", "Méthode 3", "Méthode 4", "Méthode 5", "I (Exact)"), 
       col = c("red", "blue", "green", "purple", "turquoise", "black"), 
       lty = c(1,1,1,1,1,2), cex = 0.6)
#par(mfrow = c(1,2))
hist(vect1,col='red', main = "Histogramme Méthode 1", xlab="I_chap")
abline(v=I,col='black')
hist(vect2,col='blue', main = "Histogramme Méthode 2", xlab="I_chap")
abline(v=I,col='black')
hist(vect3,col='green', main = "Histogramme Méthode 3", xlab="I_chap")
abline(v=I,col='black')
hist(vect4,col='purple', main = "Histogramme Méthode 4", xlab="I_chap")
abline(v=I,col='black')
#par(mfrow = c(1,1))
hist(vect5,col='turquoise', main = "Histogramme Méthode 5", xlab="I_chap")
abline(v=I,col='black')

#Estimation du risque quadratique de la methode
RQ = function(vect) (mean(vect)-I)^2+var(vect)
RQ_meth1 = RQ(vect1)
RQ_meth2 = RQ(vect2)
RQ_meth3 = RQ(vect3) 
RQ_meth4 = RQ(vect4) 
RQ_meth5 = RQ(vect5)
print(paste(list(RQ_meth1, RQ_meth2, RQ_meth3, RQ_meth4, RQ_meth5)))
