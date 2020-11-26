import ndjson as json
import matplotlib
matplotlib.use('qt5agg')
import pylab
import seaborn as sns
sns.set_style("whitegrid")

import pandas
import numpy as np
import scipy
#https://ipip.ori.org/New_IPIP-50-item-scale.htm

def get_confusion_matrix(scores, images):
    def shrink(name):
        return "$" + name[0] + name[-1] + "$"
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    traits_plus = [x + "+" for x in traits]
    traits_neg = [x + "-" for x in traits]
    traits_pn = [traits_plus[i // 2] if i %2 == 0 else traits_neg[i // 2] for i in range(2*len(traits))]
    ret = np.zeros((len(traits_pn), len(traits_pn)), dtype=float)
    ret = pandas.DataFrame(ret, index=list(map(shrink, traits_pn))[:], columns=list(map(shrink, traits_pn)))
    mask = np.ones((len(traits_pn), len(traits_pn)), dtype=bool)
    for i in range(len(traits_pn)):
        for j in range(len(traits_pn)):
            if abs(i - j) <= 1 and not(i != j and (i + j + 1) % 4 == 0):
                mask[i, j] = False
    for trait_i in traits:
        score = scores[trait_i].values
        for sign_i in ["+", "-"]:
            if sign_i == "+":
                score_mask = (score >= score.mean())
            else:
                score_mask = (score < score.mean())
            for trait_j in traits:
                image = images[trait_j].values
                for sign_j in ["+", "-"]:
                    if sign_j == "+":
                        image_mask = (image >= image.mean())
                    else:
                        image_mask = (image < image.mean())
                    #ret.loc[shrink(trait_i + sign_i)][shrink(trait_j + sign_j)] = (score_mask  * image_mask).sum()
                    print("AAA")
                    #ret.loc[shrink(trait_i + sign_i)][shrink(trait_j + sign_j)] = scipy.stats.spearmanr(score_mask, image_mask)[0]
                    ret.loc[shrink(trait_i + sign_i)][shrink(trait_j + sign_j)] = (score_mask * image_mask).sum() / (image_mask.sum())
    print(ret)
    sns.heatmap(ret, annot=True, cmap="Blues", fmt='.2f', mask=mask)

    pylab.gca().set_yticklabels(ret.index.values, rotation=0)
    # pylab.tight_layout(h_pad=1000)
    pylab.suptitle("Precision")
    pylab.xlabel("Trait")
    pylab.ylabel("Trait")
    pylab.gca().xaxis.tick_top()
    pylab.savefig("plots/precisionmat.eps")
    pylab.show()
    ret.to_csv("confusion.csv")


def get_confusion_matrix_join(scores, images):
    def shrink(name):
        return name[0]
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    ret = np.zeros((len(traits), len(traits)), dtype=float)
    ret = pandas.DataFrame(ret, index=list(map(shrink, traits)), columns=list(map(shrink, traits)))
    mask = np.zeros((len(traits), len(traits)), dtype=bool)
    for i in range(len(traits)):
        for j in range(len(traits)):
            if j > i :
                mask[i,j] = True

    for i, trait_i in enumerate(traits):
        score = scores[traits[i]].values
        # score = score > score.mean()
        for j, trait_j in enumerate(traits):
            if j <= i or True:
                image = images[traits[j]].values
                image = image > image.mean()

                #ret.loc[shrink(trait_i)][shrink(trait_j)] = (score * image).sum()

                ret.loc[shrink(traits[i])][shrink(traits[j])] = scipy.stats.pearsonr(score, image)[0]
                # ret.loc[shrink(traits[j])][shrink(traits[i])] = scipy.stats.pearsonr(score, image)[0]

                #ret.loc[shrink(trait_i + sign_i)][shrink(trait_j + sign_j)] = (score_mask * image_mask).sum() / (image_mask.sum())
    print(ret)
    sns.heatmap(ret, annot=True, cmap="Blues", fmt='.3f', mask=mask)
    pylab.gca().xaxis.tick_top()
    pylab.gca().set_yticklabels(ret.index.values, rotation=0)
    pylab.tight_layout(pad=2)
    # pylab.show()
    pylab.suptitle("Pearson $\\rho$")
    pylab.xlabel("Trait")
    pylab.ylabel("Trait")
    pylab.savefig('plots/ocean_pearson.eps')
    ret.to_csv("confusion.csv")




def get_pearson_matrix(scores, images, confidence_interval=0):
    traits = list(scores.columns)
    ret = np.zeros((len(traits), len(traits)))
    for i, trait_i in enumerate(traits):
        s = scores[trait_i].as_matrix()
        for j, trait_j in enumerate(traits):
            mask = np.ones_like(s, dtype=bool)
            im = images[trait_j].as_matrix()
            ret[i, j] += np.corrcoef(s[mask], im[mask])[0,1]
            if i != j:
                ret[j,i] = 0
            # ret[j, i] += np.corrcoef(scores[trait_i].as_matrix(), images[trait_j].as_matrix())[0,1] / 2.
        pylab.title(trait_i)
        pylab.bar(range(5), height=ret[i, :])
        pylab.xticks(range(5), traits, rotation='30')
        pylab.tight_layout()
        pylab.show()

    return pandas.DataFrame(ret, columns=[traits], index=[traits])

def welch_matrix(scores, images):
    traits = list(scores.columns)
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    ret = np.zeros((len(traits), len(traits)))
    for i, trait_i in enumerate(traits):
        arr_i = scores[trait_i].as_matrix()
        arr_i = (arr_i > np.median(arr_i)).astype(float) * 200 - 100
        for j, trait_j in enumerate(traits):
            arr_j = images[trait_j].values
            ret[i, j] = scipy.stats.kendalltau(arr_i, arr_j)[0]
            if i != j:
                ret[j,i] = 0
        with sns.plotting_context("notebook", font_scale=1.5):
            pylab.clf()
            cmap = sns.cubehelix_palette(100, start=0, rot=-0.25, reverse=False)
            v = ret[i, :].max() - ret[i, :].min()
            color = (99 * (ret[i, :] - ret[i, :].min()) / v).astype(int)
            for idx in range(5):
                pylab.bar(idx, height=ret[i, idx], color=cmap[color[idx]])
                pylab.text(idx, ret[i, idx] + v * (0.025 if ret[i,idx] > 0 else -0.04), traits[idx], horizontalalignment="center", fontsize=12)
            pylab.ylabel("Spearman $\\rho$")
            pylab.tight_layout()
            pylab.gca().set_xticks([])
            pylab.show()
    return pandas.DataFrame(ret, columns=[traits], index=[traits])

def binarization(scores, images, trait):
    ret = []
    scr = scores[trait].as_matrix()
    x = []
    for threshold in range(10, 100, 5):
        scrt = (scr > threshold).astype(float)
        ret.append(np.corrcoef(scrt, images[trait].values)[0, 1])
        x.append(threshold)
    return np.array(x), np.array(ret)

def save_histograms(pscores):
    with sns.plotting_context("notebook", font_scale=2.5):
        for column in pscores.columns:
            pylab.figure()
            pylab.title(str(column))
            data = pscores[column].values
            data = data - data.mean()
            data = data / data.max()
            pylab.hist(data)
            pylab.xlabel('score')
            pylab.ylabel('count')
            pylab.tight_layout(pad=0)
            pylab.xlim(-1, 1)
            pylab.savefig("plots/%s_histogram.eps" %(str(column)))
            pylab.close()

if __name__ == "__main__":
    with open("backup/personality_local.json", 'r') as infile:
        local = json.load(infile)
    with open("backup/personality_new.json", 'r') as infile:
        remote = json.load(infile)

    merged = local + remote

    scores = []
    images = []
    for datum in merged:
        scores.append(datum["scores"])
        images.append(datum["images"])

    pscores = pandas.DataFrame.from_records(scores)
    #save_histograms(pscores)
    #pscores.hist()
    pimages = pandas.DataFrame.from_records(images) * -100
    #get_confusion_matrix(pscores, pimages)
    get_confusion_matrix_join(pscores, pimages)
   # pimages.hist()
   # pylab.figure()
   # pylab.title("Thresholding")
   # rets = []
   # for i, trait in enumerate(list(pscores.columns)):
   #     pylab.subplot(3,2, i+1)
   #     x, ret = binarization(pscores, pimages, trait)
   #     rets.append(ret.copy())
   #     pylab.plot(x, ret)
   #     pylab.xlabel("threshold")
   #     pylab.ylabel("correlation")
   #     pylab.title(trait)
   # pylab.subplot(3, 2, 6)
   # pylab.plot(x, np.nanmean(rets, 0))
   # pylab.title("mean")
   # pylab.show()
   #  with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        #print(get_pearson_matrix(pscores, pimages))
        # print(welch_matrix(pscores, pimages))




