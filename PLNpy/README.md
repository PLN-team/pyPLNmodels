{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Grabbed from https://github.com/odewahn/ipynb-examples.\n",
    "\n",
    "The Markdown parser included in IPython is MathJax-aware.  This means that you can freely mix in mathematical expressions using the [MathJax subset of Tex and LaTeX](http://docs.mathjax.org/en/latest/tex.html#tex-support).  [Some examples from the MathJax site](http://www.mathjax.org/demos/tex-samples/) are reproduced below, as well as the Markdown+TeX source."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Motivating Examples\n",
    "\n",
    "---\n",
    "\n",
    "## The Lorenz Equations\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{aligned}\n",
    "\\dot{x} & = \\sigma(y-x) \\\\\n",
    "\\dot{y} & = \\rho x - y - xz \\\\\n",
    "\\dot{z} & = -\\beta z + xy\n",
    "\\end{aligned}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{aligned}\n",
    "\\dot{x} & = \\sigma(y-x) \\\\\n",
    "\\dot{y} & = \\rho x - y - xz \\\\\n",
    "\\dot{z} & = -\\beta z + xy\n",
    "\\end{aligned}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The Cauchy-Schwarz Inequality\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{equation*}\n",
    "\\left( \\sum_{k=1}^n a_k b_k \\right)^2 \\leq \\left( \\sum_{k=1}^n a_k^2 \\right) \\left( \\sum_{k=1}^n b_k^2 \\right)\n",
    "\\end{equation*}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{equation*}\n",
    "\\left( \\sum_{k=1}^n a_k b_k \\right)^2 \\leq \\left( \\sum_{k=1}^n a_k^2 \\right) \\left( \\sum_{k=1}^n b_k^2 \\right)\n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## A Cross Product Formula\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{equation*}\n",
    "\\mathbf{V}_1 \\times \\mathbf{V}_2 =  \\begin{vmatrix}\n",
    "\\mathbf{i} & \\mathbf{j} & \\mathbf{k} \\\\\n",
    "\\frac{\\partial X}{\\partial u} &  \\frac{\\partial Y}{\\partial u} & 0 \\\\\n",
    "\\frac{\\partial X}{\\partial v} &  \\frac{\\partial Y}{\\partial v} & 0\n",
    "\\end{vmatrix}  \n",
    "\\end{equation*}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{equation*}\n",
    "\\mathbf{V}_1 \\times \\mathbf{V}_2 =  \\begin{vmatrix}\n",
    "\\mathbf{i} & \\mathbf{j} & \\mathbf{k} \\\\\n",
    "\\frac{\\partial X}{\\partial u} &  \\frac{\\partial Y}{\\partial u} & 0 \\\\\n",
    "\\frac{\\partial X}{\\partial v} &  \\frac{\\partial Y}{\\partial v} & 0\n",
    "\\end{vmatrix}  \n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The probability of getting \\(k\\) heads when flipping \\(n\\) coins is\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{equation*}\n",
    "P(E)   = {n \\choose k} p^k (1-p)^{ n-k} \n",
    "\\end{equation*}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{equation*}\n",
    "P(E)   = {n \\choose k} p^k (1-p)^{ n-k} \n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## An Identity of Ramanujan\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{equation*}\n",
    "\\frac{1}{\\Bigl(\\sqrt{\\phi \\sqrt{5}}-\\phi\\Bigr) e^{\\frac25 \\pi}} =\n",
    "1+\\frac{e^{-2\\pi}} {1+\\frac{e^{-4\\pi}} {1+\\frac{e^{-6\\pi}}\n",
    "{1+\\frac{e^{-8\\pi}} {1+\\ldots} } } } \n",
    "\\end{equation*}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{equation*}\n",
    "\\frac{1}{\\Bigl(\\sqrt{\\phi \\sqrt{5}}-\\phi\\Bigr) e^{\\frac25 \\pi}} =\n",
    "1+\\frac{e^{-2\\pi}} {1+\\frac{e^{-4\\pi}} {1+\\frac{e^{-6\\pi}}\n",
    "{1+\\frac{e^{-8\\pi}} {1+\\ldots} } } } \n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## A Rogers-Ramanujan Identity\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{equation*}\n",
    "1 + \\frac{q^2}{(1-q)}+\\frac{q^6}{(1-q)(1-q^2)}+\\cdots =\n",
    "\\prod_{j=0}^{\\infty}\\frac{1}{(1-q^{5j+2})(1-q^{5j+3})},\n",
    "\\quad\\quad \\text{for $|q|<1$}. \n",
    "\\end{equation*}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{equation*}\n",
    "1 + \\frac{q^2}{(1-q)}+\\frac{q^6}{(1-q)(1-q^2)}+\\cdots =\n",
    "\\prod_{j=0}^{\\infty}\\frac{1}{(1-q^{5j+2})(1-q^{5j+3})},\n",
    "\\quad\\quad \\text{for $|q|<1$}. \n",
    "\\end{equation*}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Maxwell's Equations\n",
    "\n",
    "### Source\n",
    "\n",
    "```latex\n",
    "\\begin{aligned}\n",
    "\\nabla \\times \\vec{\\mathbf{B}} -\\, \\frac1c\\, \\frac{\\partial\\vec{\\mathbf{E}}}{\\partial t} & = \\frac{4\\pi}{c}\\vec{\\mathbf{j}} \\\\   \\nabla \\cdot \\vec{\\mathbf{E}} & = 4 \\pi \\rho \\\\\n",
    "\\nabla \\times \\vec{\\mathbf{E}}\\, +\\, \\frac1c\\, \\frac{\\partial\\vec{\\mathbf{B}}}{\\partial t} & = \\vec{\\mathbf{0}} \\\\\n",
    "\\nabla \\cdot \\vec{\\mathbf{B}} & = 0 \n",
    "\\end{aligned}\n",
    "```\n",
    "\n",
    "### Display\n",
    "\n",
    "\\begin{aligned}\n",
    "\\nabla \\times \\vec{\\mathbf{B}} -\\, \\frac1c\\, \\frac{\\partial\\vec{\\mathbf{E}}}{\\partial t} & = \\frac{4\\pi}{c}\\vec{\\mathbf{j}} \\\\   \\nabla \\cdot \\vec{\\mathbf{E}} & = 4 \\pi \\rho \\\\\n",
    "\\nabla \\times \\vec{\\mathbf{E}}\\, +\\, \\frac1c\\, \\frac{\\partial\\vec{\\mathbf{B}}}{\\partial t} & = \\vec{\\mathbf{0}} \\\\\n",
    "\\nabla \\cdot \\vec{\\mathbf{B}} & = 0 \n",
    "\\end{aligned}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Equation Numbering and References\n",
    "\n",
    "---\n",
    "\n",
    "Equation numbering and referencing will be available in a future version of IPython."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Inline Typesetting (Mixing Markdown and TeX)\n",
    "\n",
    "---\n",
    "\n",
    "While display equations look good for a page of samples, the ability to mix math and *formatted* **text** in a paragraph is also important.\n",
    "\n",
    "## Source\n",
    "\n",
    "``` \n",
    "This expression $\\sqrt{3x-1}+(1+x)^2$ is an example of a TeX inline equation in a **[Markdown-formatted](http://daringfireball.net/projects/markdown/)** sentence.  \n",
    "```\n",
    "\n",
    "## Display\n",
    "\n",
    "This expression $\\sqrt{3x-1}+(1+x)^2$ is an example of a TeX inline equation in a **[Markdown-formatted](http://daringfireball.net/projects/markdown/)** sentence.  "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
