// Copyright 2020 Romanian Institute of Science and Technology
// https://rist.ro for differential changes w.r.t. the original
// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOML_ZERO_REGULARIZED_EVOLUTION_H_
#define AUTOML_ZERO_REGULARIZED_EVOLUTION_H_

#include <memory>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "definitions.h"
#include "evaluator.h"
#include "generator.h"
#include "mutator.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "gtest/gtest_prod.h"

namespace automl_zero {

  constexpr double FIT = 1.0004;
  constexpr IntegerT PMAX = 100;

  using PSA = std::pair<std::string, std::shared_ptr<const Algorithm>>;
  
class RegularizedEvolution {
 public:
  RegularizedEvolution(
      // The compute cost of evaluating one individual.
      RandomGenerator* rand_gen,
      // Runs up to this many total individuals.
      IntegerT population_size,
      IntegerT tournament_size,
      // How frequently to print progress reports.
      IntegerT progress_every,
      Generator* generator,
      Evaluator* evaluator,
      // The mutator to use to perform all mutations.
      Mutator* mutator);
  RegularizedEvolution(
      const RegularizedEvolution& other) = delete;
  RegularizedEvolution& operator=(
      const RegularizedEvolution& other) = delete;
  // Initializes the algorithm. Returns the number of individuals evaluated in
  // this call.
  double Init();
  // Runs for a given amount of time (rounded up to the nearest generation) or
  // for a certain number of train steps (rounded up to the nearest generation),
  // whichever is first. Assumes that Init has been called. Returns the number
  // of train steps executed in this call.
  IntegerT Run(IntegerT max_train_steps, IntegerT max_nanos, double min_fitness);
  // Returns the CUs/number of individuals evaluated so far.
  // Returns an exact number.
  IntegerT NumIndividuals() const;
  // The number of train steps executed.
  IntegerT NumTrainSteps() const;
  // Returns a random serialized Algorithm in the population and its fitness.
  std::shared_ptr<const Algorithm> Get(double* fitness);
  // Returns the best serialized Algorithm in the population and its worker
  // fitness.
  std::shared_ptr<const Algorithm> GetBest(double* fitness);
  IntegerT PopulationSize() const;
  void PopulationStats(double* pop_mean, double* pop_stdev,
      std::shared_ptr<const Algorithm>* pop_best_algorithm,
      double* pop_best_fitness) const;

 private:
  FRIEND_TEST(RegularizedEvolutionTest, TimesCorrectly);
  friend IntegerT PutsInPosition(const Algorithm&, RegularizedEvolution*);
  friend IntegerT EvaluatesAndPutsInPosition(const Algorithm&, RegularizedEvolution*);
  friend bool PopulationsEq(const RegularizedEvolution&, const RegularizedEvolution&);

  std::string STAMP_; std::ofstream rf_, af_, of_; IntegerT fs_;
  double best_fit_, cull_fit_; int sc_, epc_; bool LT(int c, double prev_fit);
  bool Cull(double prev_fit, int c,int& pcA,int& pcB,int& pcC, double min_fitness);
  double CF(IntegerT K, bool cut,int& pcA,int& pcB,int& pcC, double min_fitness);   
  double Kick(double prev_fit, double min_fitness, IntegerT max_nanos,
	      IntegerT max_train_steps, int& pcA, int& pcB, int& pcC);
  bool AveMaria(double prev_fit); bool HailMary(double prev_fit);
  bool NextMaria(double prev_fit); bool NextMary(double prev_fit);
  void resEPC(); bool resCF(); void resPOP(int new_pop); IntegerT CP(int c); 
  double MeanFit(); double LastTry(int c); double Fetch(); double NextFetch();  
  double Fetch(bool forced, IntegerT fs); double NextFetch(bool forced);
  double NextFetch(bool forced, IntegerT fs); double Fetch(bool forced);
  double RunHybrid(IntegerT max_train_steps, IntegerT max_nanos, double min_fitness);
  double RunSw(double min_fitness) { RunV2(min_fitness); RunV1(min_fitness); return RunV0(min_fitness); };
  bool RunSp(double min_fitness, double prev_fit) { if(RunSw(min_fitness)>=FIT*prev_fit) { 
      of_ << "[RS],"; resEPC(); of_.flush(); return true; } return AveMaria(prev_fit); };
  double RunDMH(double min_fitness,int& pcA,int& pcB,int& pcC);  
  double Run3H(double min_fitness,int& pcA,int& pcB,int& pcC);  
  double Run3Hw(double min_fitness,int& pcA,int& pcB,int& pcC);  
  double RunHw(double min_fitness,int& pcA,int& pcB,int& pcC);
  double RunM(double min_fitness,int& pcA,int& pcB,int& pcC);
  double RunMw(double min_fitness,int& pcA,int& pcB,int& pcC);
  bool RunMp(double min_fitness,int& pcA,int& pcB,int& pcC,double prev_fit);
  bool RunHp(double min_fitness,int& pcA,int& pcB,int& pcC,double prev_fit);  
  bool CLAR(int c, double min_fitness,int& pcA,int& pcB,int& pcC,double prev_fit);   
  double RunV1(double min_fitness); double RunV1w(double min_fitness);
  double RunV2(double min_fitness); double RunV2w(double min_fitness);
  double RunV0(double min_fitness); double RunV0w(double min_fitness);
  double MaybePrintProgress(bool forced); void DimUp(double nt);
  void PrintFit(); double NextMax(int j); double FullMax(int j);    
  std::shared_ptr<const Algorithm> ReEvaluate(bool cut, IntegerT K, bool& change);
  bool Push(double prev_fit, IntegerT new_pop); void Pull(IntegerT K, bool cut);
  
  void InitAlgorithm(std::shared_ptr<const Algorithm>* algorithm) {
    *algorithm = std::make_shared<Algorithm>(generator_->TheInitModel()); } ;
  void UpdateIS(int k) { num_individuals_+=k; epoch_secs_=absl::GetCurrentTimeNanos()/kNanosPerSecond; };
  void FirstExecute(int i){ FirstExecute(i,fs_); }; void NextExecute(int j){ NextExecute(j,fs_); }; 
  void FirstExecute(int i, IntegerT fs) { fitnesses_[i] = Execute(algorithms_[i], fs); };
  void NextExecute(int j, IntegerT fs) { next_fitnesses_[j] = Execute(next_algorithms_[j], fs); }; 
  //double Execute(std::shared_ptr<const Algorithm> algorithm) {return Execute(algorithm, fs_);};
  double Execute(std::shared_ptr<const Algorithm> algorithm, IntegerT fs) {
    return evaluator_->Evaluate(*algorithm, fs); };
  std::shared_ptr<const Algorithm> BestFitnessTournament();  
  //std::shared_ptr<const Algorithm> BestFitnessTournament(int ps);
  std::shared_ptr<const Algorithm> NextFitnessTournament(int ps);
  void Mutate(std::shared_ptr<const Algorithm>* algorithm);
  void FirstMutate(int i) {Mutate(&algorithms_[i]);};
  void NextMutate(int j) {Mutate(&next_algorithms_[j]);};  
  void FullSelect(std::shared_ptr<const Algorithm>* algorithm) {
    *algorithm = BestFitnessTournament(); Mutate(algorithm); };
  //void FirstSelect(std::shared_ptr<const Algorithm>* algorithm, int ps);
  //void FirstSelect(int i) {NextSelect(algorithms_[i], population_size_)};  
  void NextSelect(std::shared_ptr<const Algorithm>* algorithm, int ps) {
    ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
    *algorithm = NextFitnessTournament(ps); Mutate(algorithm); };
  void NextSelect(int j) {NextSelect(&next_algorithms_[j], population_size_);};
  void SecondFirst(int j, int i) { next_algorithms_[j] = algorithms_[i]; next_fitnesses_[j] = fitnesses_[i]; };
  void FirstFirst(int j, int i) { algorithms_[j] = algorithms_[i]; fitnesses_[j] = fitnesses_[i]; };  

  Evaluator* evaluator_;
  RandomGenerator* rand_gen_;
  const IntegerT start_secs_;
  IntegerT epoch_secs_;
  IntegerT epoch_secs_last_progress_;
  IntegerT num_individuals_last_progress_;
  IntegerT tournament_size_;
  const IntegerT progress_every_;
  bool initialized_;
  Generator* generator_;
  Mutator* mutator_;

  // Serializable components.
  std::map<std::string, std::shared_ptr<const Algorithm>> dict_;
  std::pair<std::map<std::string, std::shared_ptr<const Algorithm>>::iterator,bool> ret_;

  int cf_ = 1; // number of preserved best algorithms
  int NP = 2; // number of available CPUs, e.g., 8
  int HNP = NP/2; // half the number of available CPUs
  // it is recommended that population_size_ is NP+cf_
  IntegerT population_size_ = NP+cf_;
  IntegerT init_pop_ = population_size_;
  IntegerT min_pop_ = population_size_;
  IntegerT max_pop_ = population_size_;  
  std::vector<std::shared_ptr<const Algorithm>> algorithms_;
  std::vector<std::shared_ptr<const Algorithm>> next_algorithms_;    
  std::vector<double> fitnesses_;
  std::vector<double> next_fitnesses_;    
  IntegerT num_individuals_;
};

}  // namespace automl_zero

#endif  // AUTOML_ZERO_REGULARIZED_EVOLUTION_H_
