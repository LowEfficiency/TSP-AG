import os
import statistics
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.AG import AG
from src.TSP import TSP
from src.Individuo import Individuo

class Experimento:
    def __init__(self, caminho_instancia, n_execucoes=20, n_geracoes=20, pasta_base="saidas"):
        self.tsp = TSP(caminho_instancia)
        self.n_execucoes = n_execucoes
        self.n_geracoes = n_geracoes
        
        self.pasta_saida = self._configurar_pasta_saida(caminho_instancia, pasta_base)
       
        self.fatores = {
            'populacao' : [50, 100],
            'taxa_cruzamento' : [0.8, 0.9],
            'taxa_mutacao' : [0.01, 0.05],
            'operador' : ['OX', 'PMX']
        }
        
        self.dados_tabela = [] 
        self.dados_brutos_boxplot = []
        self.convergencia_global = {}
        self.melhor_absoluto = None

    def _configurar_pasta_saida(self, caminho_instancia, pasta_base):
        nome_arquivo = os.path.basename(caminho_instancia)
        nome_instancia, _ = os.path.splitext(nome_arquivo)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pasta_saida = os.path.join(pasta_base, f"{nome_instancia}_{timestamp}")
        os.makedirs(pasta_saida, exist_ok=True)
        return pasta_saida
        
    def _executar_configuracao(self, config, id_config):
        """Executa as N rodadas para uma configuração específica e retorna as métricas."""
        fitness_por_execucao = []
        matriz_convergencia = np.zeros((self.n_execucoes, self.n_geracoes))
        
        melhor_rota_config = None          
        melhor_fitness_config = float('inf')
        
        for rodada in range(self.n_execucoes):
            ga = AG(
                tamanho_pop=config['populacao'],
                taxa_cruzamento=config['taxa_cruzamento'],
                taxa_mutacao=config['taxa_mutacao'],
                operador=config['operador'],
                tsp_instancia=self.tsp
            )
            
            for gen in range(self.n_geracoes):
                ga.evolucao()
                melhor_da_gen = min(ga.populacao, key=lambda ind: ind.fitness)
                matriz_convergencia[rodada, gen] = melhor_da_gen.fitness
                
                if self.melhor_absoluto is None or melhor_da_gen.fitness < self.melhor_absoluto.fitness:
                    self.melhor_absoluto = Individuo(melhor_da_gen.rota[:], self.tsp)
            
            melhor_final = matriz_convergencia[rodada, -1]
            fitness_por_execucao.append(melhor_final)
            
            # Atualiza o melhor indivíduo da configuração atual
            melhor_ind_rodada = min(ga.populacao, key=lambda ind: ind.fitness)
            if melhor_ind_rodada.fitness < melhor_fitness_config:
                melhor_fitness_config = melhor_ind_rodada.fitness
                melhor_rota_config = melhor_ind_rodada.rota[:]
                
            # Salva os dados brutos para o Boxplot
            self.dados_brutos_boxplot.append({
                'Config': id_config,
                'Fitness': melhor_final,
                'Operador': config['operador']
            })
            
        convergencia_media = np.mean(matriz_convergencia, axis=0)
        return fitness_por_execucao, melhor_rota_config, convergencia_media

    def rodar(self):
        """Gerencia o loop principal do experimento e compila os resultados."""
        combinacoes = list(product(*self.fatores.values()))
        coluna_fatores = list(self.fatores.keys())
        
        print(f"Iniciando Teste Fatorial: {len(combinacoes)} configs x {self.n_execucoes} execuções.")
        
        for i, valores in enumerate(combinacoes):
            config = dict(zip(coluna_fatores, valores))
            id_config = f"C{i+1}"
            print(f"Executando {id_config} -> População: {config['populacao']} | Cruzamento: {config['taxa_cruzamento']} | Mutação: {config['taxa_mutacao']} | Operador: {config['operador']}")
            
            fitness_por_execucao, melhor_rota, convergencia_media = self._executar_configuracao(config, id_config)
            
            self.convergencia_global[id_config] = convergencia_media
            
            self.dados_tabela.append({
                'ID': id_config,
                'Operador': config['operador'],
                'Pop': config['populacao'],
                'TX_C': config['taxa_cruzamento'],
                'TX_M': config['taxa_mutacao'],
                'Melhor': min(fitness_por_execucao),
                'Pior': max(fitness_por_execucao),
                'Média': statistics.mean(fitness_por_execucao),
                'Mediana': statistics.median(fitness_por_execucao),
                'Desvio Padrão': statistics.stdev(fitness_por_execucao),
                'Vetor da Rota (Melhor)': str(melhor_rota)
            })

        df_tabela = pd.DataFrame(self.dados_tabela)
        caminho_csv = os.path.join(self.pasta_saida, "resultados_fatorial_completo.csv")
        df_tabela.to_csv(caminho_csv, index=False, sep=';')
        
        return df_tabela
    
    def gerar_graficos(self):
        """Chama as funções de geração de todos os gráficos."""
        df_box = pd.DataFrame(self.dados_brutos_boxplot)
        self._plot_boxplot(df_box)
        self._plot_barras(df_box)
        self._plot_convergencia()

    def _plot_boxplot(self, df_box):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Config', y='Fitness', hue='Operador', data=df_box)
        plt.title("Boxplot: Dispersão por Configuração")
        plt.savefig(os.path.join(self.pasta_saida, "boxplot_final.png"))
        plt.close()

    def _plot_barras(self, df_box):
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Config', y='Fitness', data=df_box, capsize=.1)
        plt.title("Média e Desvio Padrão por Configuração")
        plt.savefig(os.path.join(self.pasta_saida, "barras_media.png"))
        plt.close()

    def _plot_convergencia(self):
        plt.figure(figsize=(12, 6))
        for id_config, valores in self.convergencia_global.items():
            plt.plot(valores, label=id_config)
        plt.title("Evolução da Convergência (Média das 50 execuções)")
        plt.xlabel("Geração")
        plt.ylabel("Melhor Fitness")
        plt.legend(loc='upper right', ncol=2)
        plt.savefig(os.path.join(self.pasta_saida, "convergencia_final.png"))
        plt.close()
