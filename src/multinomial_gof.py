##########################################
# ADAPTED FROM THE ORIGINAL VERSION met.py
# original text header, license etc. follow
##########################################

"""Perform exact tests of a (site or test) distribution of multinomial count data against
a distribution of equivalent ordered multinomial count data from another (reference or control)
data set.  Both two-sided and one-sided tests can be performed.  One-sided tests require
that categories be ordered.
"""


# met.py
# Multinomial Exact Tests
#
# PURPOSE:
#	Define an object and methods to represent categorical data from two populations
#	and to allow exact multinomial tests of one population against the other.
#	A practical example of relevant data (and the motivation for writing this module) is
#	the number of samples found at different benthic successional stages at site and
#	reference locations.  These categorical data are ordered, and so a one-sided exact
#	multinomial test can be applied.  A method for two-sided exact multinomial tests
#	is also implemented.
#
# USAGE:
#   This module implements a class that is used to store and manipulate related sets of
#	multinomial data (e.g., sediment profile image [SPI] successional stage data
#   for reference and site locations).  Computation of the p value for a test of site data
#	against reference conditions  requires two Python statements (in addition to the
#   module import statement): one to instantiate the Multinom data object, and one to
#   request that it calculate an exact p value.
#   These statements, if issued at the Python interpreter, might look like this:
#       >>> from met import Multinom
#       >>> my_site = Multinom([6, 2, 1], [86, 24, 15])
#       >>> my_site.onesided_exact_test()
#   The 'onesided_exact_test()' and 'twosided_exact_test()' methods are the principal methods
#	of the Multinom object.  Both methoods return the p value for the exact test.
#   This p value is also stored in the Multinom object after an exact test is carried out
#   (attribute 'p_value').  The number of extreme cases found is also retained following
#   a calculation (attribute 'n_extreme_cases').  The cases themselves may optionally be saved
#   during a calculation (through an optional argument to the '..._exact_test()' methods);
#	these are saved as a list of lists, where each of the component lists is one distribution
#	of site sample counts across SPI stages that is more extreme than the reference area.
#
# NOTES:
#	1. The calculation is performed by summing the multinomial probabilities for the
#		observed distribution and all distributions that are more extreme than observed.
#	2. For one-sided tests, "more extreme" means that one or more observations is shifted
#		from a more-reference-like category (e.g., higher successional stage) to a lower one.
#	3. For two-sided tests, "more extreme" means that the probability of an alternative
#		arrangement of site data has a lower probability than the observed arrangement.
#	4. To carry out one-sided tests, categories should be listed from most reference-like
#		to least reference-like.  (For benthic successional data, Stage 3 should be listed
#		first, and stage 1 listed last.)
#
# AUTHOR:
#	Dreas Nielsen (RDN)
#   dnielsen@integral-corp.com
#
# COPYRIGHT AND LICENSE:
#	Copyright (c) 2009, 2019, R. Dreas Nielsen
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# The GNU General Public License is available at <http://www.gnu.org/licenses/>
#
# HISTORY:
#		 Date		 Revisions
#		----------	------------------------------------------------------------------------
#		2007		Implemented 'listprod()', 'fact()', 'multinom_p()',
#					and 'obs_to_probs()'.  Began code for 'extrema()' and other supporting
#					routines.  RDN.
#		1/26/2008	Wrote 'subset()' and 'redistrib()'.  RDN.
#		1/27/2008	Revised 'subset()', completely re-created 'extrema()', and eliminated
#					other incomplete (and now unnecessary) code.  RDN.
#       3/11/2008   Revised init call to '__reset()'.  RDN.
#		6/16/2008	Began 'all_multinom_cases()'.  RDN.
#       1/10/2009   Added usage and function documentation.  RDN.
#		1/11/2009	Renamed module and class for generality.  RDN.
#       1/13/2009   Completed 'all_multinom_cases()'.  This could go in a sub-module because
#                   it is to support further analysis, and is not strictly necessary for the
#                   Multinom class.  Renamed 'Multinom.exact_test()' to 'Multinom.onesided_exact_test()'.
#                   Added 'onesided_exact_likelihood()'.  RDN.
#		1/14/2009	Added 'fill_zeroes' and 'fill_factor' arguments to 'Multinom.onesided_exact_test()'
#					and 'onesided_exact_likelihood()' to eliminate the (random) zeroes problem
#					in the distribution of (typically a small number of) reference area observations.
#					A default fill factor is set to 10 (this is the value by which all reference
#					observations are multiplied before adding 1 to eliminate the zeroes)--this
#					value could be altered, augmented, or replaced by a heuristic based on the
#					total number of reference area measurements, and the distribution among them
#					(i.e., the more reference area measurements there are, the more certain you
#					are likely to be that an observed zero for some stage is a true zero).  RDN.
#		1/18/2004	Changed to normalize likelihoods to the maximum rather than the sum, for
#					compatibility with individually calculated p values
#					(in 'onesided_exact_likelihood()'), and added the normalized likelihood
#					to the output of 'onesided_exact_likelihood()'.  Corrected set/reset of ref_probs
#					in 'onesided_exact_test()' when zero filling is used. RDN.
#		1/19/2009	Added 'ref_samples' argument to 'onesided_exact_likelihood()'.  RDN.
#		1/20/2009	Added 'fillzeroes()' and modified other routines to use it.  RDN.
#		1/21/2009	Added 'twosided_exact_test()' and 'twosided_exact_likelihood().  RDN.
#		1/25/2009	Edited documentation in source.  RDN.
#		2019-09-27	Modified docstrings.  Changed assignment of total counts in
#					'twosided_exact_test()'.  RDN.
#		2019-09-29	Modified to run under Python 3 as well as 2.  Version no. 1.0.0  RDN.
#============================================================================================
import numpy as np
import copy


class MultinomialGOF(object):

    def __init__(self, observed, null_proportions):
        self.observed = observed
        self.null_proportions = null_proportions
        self.p_value = 0.0
        self.n_cats = observed.shape[0]
        self.n_counts = np.sum(observed)
        self.n_factorial = np.math.factorial(self.n_counts)
        self.__curr_mod_src = None

    def multinomial_probability(self, observed, null_proportions):
        first_term = self.n_factorial
        for n in range(0, self.n_cats):
            first_term /= np.math.factorial(observed[n])

        return first_term * np.prod(null_proportions ** observed)

    def new_sink_set(self, sink_set):
        """Internal callback routine.
        """
        new_case = np.hstack((self.__curr_mod_src, sink_set))
        self.p_value += self.multinomial_probability(new_case, self.null_proportions)

    def redistrib(self, counts, sublist, accumlist):
        """Create lists of all combinations of 'counts' redistributed over 'sublist',
        which is a simple list of integers (counts).  Execute 'new_sink_set()'
        with each combination.
        """
        if sublist.shape[0] == 1:
            newaccum = np.hstack((accumlist, counts + sublist[0]))
            self.new_sink_set(newaccum)
        else:
            for i in range(counts+1):
                newaccum = np.hstack((accumlist, i + sublist[0]))
                self.redistrib(counts-i, sublist[1:], newaccum)

    def new_src_set(self, src_set):
        """Callback routine for find_subset.
        """
        self.__curr_mod_src = [self.observed[i] - src_set[i] for i in range(src_set.shape[0])]
        n_to_move = np.sum(src_set).astype(np.int64)
        self.redistrib(n_to_move, self.observed[src_set.shape[0]:], [])

    def subset(self, sublist, accumlist):
        """Find all combinations of elements of 'sublist' (which is a simple list
        of integers [counts]).  Each combination has a length equal to 'sublist',
        and zeroes may appear in any position (category) except the last.  Execute
        'new_src_set()' with each combination.
        """
        if len(sublist) == 0:
            self.new_src_set(accumlist)
        else:
            if sublist[0] == 0:
                newaccum = np.hstack((accumlist, 0))
                self.subset(sublist[1:], newaccum)
            else:
                if sublist.shape[0] == 1:
                    for i in range(sublist[0]):
                        newaccum = np.hstack((accumlist, i+1))
                        self.subset(sublist[1:], newaccum)
                else:
                    for i in range(sublist[0]+1):
                        newaccum = np.hstack((accumlist, i))
                        self.subset(sublist[1:], newaccum)

    def multinom_cases(self, counts, categories, items):
        """Add to the list of cases a new case consisting of the 'counts' list, after
        distributing 'items' items over 'categories' remaining categories.
        """
        if categories == 1:
            # There's only one category left to be filled, so put all remaining items in it.
            counts.append(items)
            p = self.multinomial_probability(counts, self.null_proportions)
            if p <= self.ref_p:
                self.p_value += p
        elif items == 0:
            # There are no more items, so put 0 in all remaining categories
            for n in range(categories):
                counts.append(0)
            p = self.multinomial_probability(counts, self.null_proportions)
            if p <= self.ref_p:
                self.p_value += p
        else:
            for n in range(items+1):
                newcounts = copy.copy(counts)
                newcounts.append(n)
                self.multinom_cases(newcounts, categories-1, items-n)

    def all_multinom_cases(self, categories, items):
        """Returns a list of all multinomial combinations (each a list) of 'items' items distributed
        in all possible ways over 'categories' categories."""
        if categories==0 and items>0:
            raise MultinomError("Can't distribute %d items over 0 cases." % items)
        self.multinom_cases([], categories, items)

    def random_perm_test(self, n_perms=1000):

        p_distribution = np.zeros(n_perms)
        new_draws = np.random.multinomial(self.n_counts, self.null_proportions, size=(n_perms))
        for perm in range(0, n_perms):
            p_distribution[perm] = self.multinomial_probability(new_draws[perm, :], self.null_proportions)

        self.p_value = np.count_nonzero(p_distribution <= self.ref_p) / n_perms
        return self.p_value


    def onesided_exact_test(self):
        self.p_value = self.multinomial_probability(self.observed, self.null_proportions)
        for i in range(self.n_cats - 1):
            if self.observed[i] > 0:
                orig_src = self.observed[0:i+1]
                self.subset(orig_src, np.array([]))

        return self.p_value

    def twosided_exact_test(self, p_cut=1.):

        ref_p = self.multinomial_probability(self.observed, self.null_proportions)
        self.ref_p = ref_p
        self.all_multinom_cases(self.n_cats, self.n_counts)

        return self.p_value
