# Introduction

## Monte Carlo Simulations for Direct Air Capture (DAC)

A key challenge for DAC lies in identifying materials that exhibit strong CO₂ affinity and high selectivity over H₂O. To address this, Monte Carlo (MC) simulations such as Widom insertion and Grand Canonical Monte Carlo (GCMC) are used to determine gas adsorption properties of materials by calculating the heat of adsorption ($Q_{st}$)
 and Henry’s law coefficient ($K_H$). These ensemble-averaged properties enable a more comprehensive evaluation than relying solely on single configuration properties such as adsorption energies at the most likely stable binding sites.

The Widom insertion method is well-suited for DAC under the condition of low CO$_{2}$ concentration, where modeling a dilute system is applicable. The simulation involves numerous individual calculations to build the ensemble, where single gas molecules are randomly inserted into MOF frameworks to compute the ensemble-averaged interaction energy. The interaction energy $U_{int}$ is determined from:

$$ U_{int} = U(system) - U(MOF) - U(gas) $$

where $U(system)$ is the energy of the MOF with the inserted gas molecule, $U(MOF)$ is the energy of the MOF framework, and $U(gas)$ is the energy of the gas molecule. The heat of adsorption ($Q_{st}$) is calculated as the average interaction energy of the inserted gas molecule with the MOF framework. The Henry’s law coefficient ($K_{H}$) is proportional to the averaged existence probability of gas molecule within framework following Boltzmann distribution.

Since these values ($Q_{st}$ and $K_{H}$) are changed according to the temperature, the additional term or multiplier to consider the temperature contribution was used.
Therefore, the properties of the system can be calculated as follows:

$$ Q_{st} = \frac{\langle U_{int} \exp(-\beta U_{int}) \rangle}{\langle \exp(-\beta U_{int}) \rangle} - {k_\beta}T$$

$$ K_H = \beta \langle \exp(-\beta U_{int}) \rangle $$

where $\beta$ represents the inverse of the thermal energy, defined as $β=1/(k_\beta T)$, where $k_\beta$ is the Boltzmann constant and $T$ is the temperature. Without the contribution from the thermal term ${k_\beta}T$ in $Q_{st}$, the value represents the ensemble-averaged interaction energy, denoted as $∆h_i$. For convenience, the magnitude of $Q_{st}$ is represented with an inverse sign in the original paper.

## Machine Learning Force Field (MLFF)

While the Widom insertion is a powerful simulation technique, achieving statistically robust results necessitates extensive computations, often requiring at least 10,000 samples for a converged ensemble average. This situation is especially important for porous materials (e.g. metal-organic fraemworks (MOFs)) which have large cavity to accomodate gas molecule. Although ab initio methods like Density Functional Theory (DFT) provide high accuracy, they are computationally infeasible for numerous iterations of repeated energy calculations which are accompanied with the Widom insertion MC simulation. Consequently, most MC simulations on porous materials rely on classical force fields, such as universal force fields (UFF) combined with point charges (e.g. DDEC). Despite its computational efficientness, classical force fields are known to show trouble in accurately simulate the meticulous interaction between framework and gas molecule, particularly in capturing MOF/CO₂ and MOF/H₂O interaction energies. High polarity originated from polar bond within those molecules foster tricky chemical environment that cannot be easily simulated by static classical force field. Such limitations underscore the need for more advanced approaches to maintain both speed and accuracy.

```{figure} ../assets/methods_comparison.svg
:width: 500px
:name: fig-methods
:align: center
```

To address these challenges, `DAC-SIM` introduces a transferable machine learning force field, MACE-DAC, specifically designed for accurate modeling of CO₂ and H₂O interactions within MOFs. This force field is developed by finetuning the foundation model MACE-MP-0, pretrained on inorganic materials, tailored for DAC applications. These advancements enabled us to perform large-scale, ab initio-level simulations at speeds traditionally associated with classical force fields.

```{figure} ../assets/toc.svg
:width: 500px
:name: fig-toc
:align: center
```

## DAC-SIM package

The `DAC-SIM` package streamlines the entire simulation workflow, integrating MACE-DAC force fields into Widom insertion simulations. The package automates processes such as setting up simulations, executing computations, and analyzing results, making it accessible for a wide range of users. In this workflow, target gas molecules (CO₂ or H₂O) are randomly inserted into the MOF framework to compute key ensemble-averaged properties, including the heat of adsorption ($Q_{st}$) and Henry’s law coefficient ($K_H$). The package also provides a user-friendly interface for users to customize simulation parameters, such as temperature, the number of insertion steps for high throughput simulations purposes. Moreover, user custom MLFFs based on MACE foundtion model can be easily adopted by replacing MACE-DAC.

```{figure} ../assets/workflow.svg
:width: 1000px
:name: fig-workflow
:align: center
```

This package also supports molecular dynamics (MD) simulations and geometry optimization for MOF structures. The MD simulations allow users to study the dynamic behavior of MOFs under different conditions and consider the kinetic behavior of inserted gas molecule within framework.

The geometry optimization feature enables users to relax MOF structures to their equilibrium state. These features are essential for understanding the structural stability and dynamic behavior of MOFs, which are crucial for DAC applications.

Let's get started with the installation and usage of the `DAC-SIM` package. 🚀
