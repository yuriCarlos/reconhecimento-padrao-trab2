# reconhecimento-padrao-trab3

 Faces recognition example using eigenfaces and SVMs
 
 
 The dataset used in this example is a preprocessed excerpt of the
 "Labeled Faces in the Wild", aka LFW_:
 
   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
 
 .. _LFW: http://vis-www.cs.umass.edu/lfw/
 
 Expected results for the top 5 most represented people in the dataset:
 
 ================== ============ ======= ========== =======
                    precision    recall  f1-score   support
 ================== ============ ======= ========== =======
      Ariel Sharon       0.67      0.92      0.77        13
      Colin Powell       0.75      0.78      0.76        60
   Donald Rumsfeld       0.78      0.67      0.72        27
     George W Bush       0.86      0.86      0.86       146
 Gerhard Schroeder       0.76      0.76      0.76        25
       Hugo Chavez       0.67      0.67      0.67        15
        Tony Blair       0.81      0.69      0.75        36
 
       avg / total       0.80      0.80      0.80       322
 ================== ============ ======= ========== =======
 


Yuri Carlos Bonifácio Neves - 1609670

 Projeto 03 - Reconhecimento de Faces para a disciplina de Reconhecimento de Padrões em Imagens (2017/1)


   O presente trabalho procura classificar as faces de identidades famosas
por meio de eigen faces, tendo suas dimensionalidades reduzidas a partir
do algoritmo PCA e, posteriormente, classificadas utilizando 5 classificadores,
sendo eles:
   DecisionTree, Knn, Naive Bayes, LDA e SVM

   Primeiramente é executado uma otimização na escolha dos parâmetros para cada um 
dos classificadores por meio do GridSearchCv. As melhores escolhas são usadas no 
processo de treino e classificação, e estão discriminadas a baixo 
   Após este processo, é executado o PCA, a fim de reduzir a dimensionalidade das
eigenFaces.
   Como os classificadores são diferentes e possuem características de funcionamento
diferentes é executado uma busca a fim de encontrar qual o melhor número de 
componentes para cada um destes classificadores. Os valores selecionados foram:

   DecisionTree - 45
   Naive Bayes  - 75
   Knn          - 65
   LDA          - 80
   SVM          - 80

   Então é feita a classificação com cada um dos classificadores com os parâmetros
e valores de componentes obitidos.

   Foi usada uma divisão de 75% dos dados para treino e 25% para teste.

   O Decision Tree foi o classificador com o pior resultado, como já era de se esperar,
uma vez que os dados são pouco categóricos. Seus resultados médios foram:
                  precision    recall      f1         support
                  0.53         0.52        0.52       322

   No Naive Bayes, por não possuir uma grande quantidade de parâmetros, a fim de 
otimizar os resultados, foi definido como prior as proporções de cada classe,
ou seja, razão entre a quantidade de amostras que uma classe possuia e a quantidade
de amostras total. Aparentemente a definição de priors dessa maneira não foi
muito eficiente, já que não influenciou muito nos resultados.

Antes de se definir as priors
                 precision    recall      f1         support
                 0.78         0.77        0.76       322

Depois de se definir as priors
                 precision    recall      f1         support
                 0.77         0.75        0.75       322

   O Knn seguiu com resultados não muito bons. Suspeito que seja devido ao número
de amostras. possivelmente, se tivessemos mais amostras o Knn teria um melhor 
desempenho. Segue os resultados:

                 precision    recall      f1         support
                 0.72         0.72        0.71       322

   LDA teve resultados muito bons, bem próximos do SVM. Infelizmente, mesmo após 
o processo do LDA, as classes ainda ficam meio misturadas, provavel que por este
motivo os resultados não sejam melhores.
   Em termos de carga computacional, por sua proximidade de resultados com o SVM,
pode ser uma boa opção escolher usar o LDA em uma aplicação como essa em vez 
do SVM.

                 precision    recall      f1         support
                 0.80         0.80        0.80       322

   O SVM obteve os melhores resultados, sendo eles:

                 precision    recall      f1         support
                 0.87         0.86        0.86       322
