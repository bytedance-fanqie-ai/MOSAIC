<h3 align="center">
    <img src="assets/logo.png" alt="Logo" style="vertical-align: middle; width: 40px; height: 40px;">
    MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement
</h3>

<p align="center"> 
<a href="https://bytedance-fanqie-ai.github.io/MOSAIC/"><img alt="Build" src="https://img.shields.io/badge/Project%2520Page-MOSAIC-blue?style=flat&logo=aaa&label=Project%20Page"></a> 


<!-- ><p align="center"> <span style="color:#137cf3; font-family:Gill Sans">Dong She<sup>*</sup></span>, <span style="color:#137cf3; font-family:Gill Sans">Siming Fu<sup>*</sup></span>, <span style="color:#137cf3; font-family:Gill Sans">Mushui Liu<sup>*</sup></span>,<span style="color:#137cf3; font-family:Gill Sans">Qiaoqiao Jin<sup>*</sup></span>, <span style="color:#137cf3; font-family:Gill Sans">Hualiang Wang<sup>*</sup></span>,  <br> <span style="color:#137cf3; font-family:Gill Sans">Mu Liu</span>, <span style="color:#137cf3; font-family:Gill Sans">Jidong Jiang<sup>+</sup></span></a> <br>
><span style="font-size: 16px">Fanqie AI Team, ByteDance</span></p> -->

<p align="center"> 
  <a href="https://scholar.google.com/citations?user=DcVoflUAAAAJ&hl=zh-CN&oi=ao"><span style="color:#137cf3; font-family:Gill Sans">Dong She<sup>*</sup></span></a>, 
  <a href="https://scholar.google.com/citations?user=tql_Zc4AAAAJ&hl=zh-CN&oi=ao"><span style="color:#137cf3; font-family:Gill Sans">Siming Fu<sup>*</sup></span></a>, 
  <a href="https://scholar.google.com/citations?user=-WUyWpMAAAAJ&hl=zh-CN&oi=ao"><span style="color:#137cf3; font-family:Gill Sans">Mushui Liu<sup>*</sup></span></a>, 
  <a href="https://scholar.google.com/citations?user=zWQf0XcAAAAJ&hl=zh-CN&oi=ao"><span style="color:#137cf3; font-family:Gill Sans">Qiaoqiao Jin<sup>*</sup></span></a>, 
  <a href="https://scholar.google.com/citations?user=4lzd8NsAAAAJ&hl=zh-CN&oi=ao"><span style="color:#137cf3; font-family:Gill Sans">Hualiang Wang<sup>*</sup></span></a>,  
  <br> 
  <span style="color:#137cf3; font-family:Gill Sans">Mu Liu</span></a>, 
  <span style="color:#137cf3; font-family:Gill Sans">Jidong Jiang<sup>+</sup></span></a> 
  <br>
  <span style="font-size: 16px">Fanqie AI Team, ByteDance</span>
</p>




## 🔥 News
- [30/09/2025] 🔥 Release training/inference codes and [models](https://huggingface.co/ByteDance-FanQie/MOSAIC)(resolution 512x512). The vision of resolution 1024x1024 is coming soon.
- [02/09/2025] The [arXiv paper](https://arxiv.org/abs/2509.01977v1) of MOSAIC is released.
- [08/20/2025] The [project page](https://bytedance-fanqie-ai.github.io/MOSAIC/) of MOSAIC is released.

## 📖 Introduction
<p align="center">
<img src="assets/teaser.png" width=95% height=95% 
class="center">
</p>
We present <b>MOSAIC</b>, a representation-centric framework that rethinks multi-subject generation through explicit semantic correspondence and orthogonal feature disentanglement. Our key insight is that multi-subject generation requires precise semantic alignment at the representation level—knowing exactly which regions in the generated image should attend to which parts of each reference. 
<p align="center">
<img src="assets/pipeline.png" width=95% height=95% 
class="center">
</p>
MOSAIC introduces two key supervisions: (1) <b>Semantic Correspondence Attention Loss</b> (blue region) enforces precise point-to-point alignment between reference tokens and their corresponding locations
in the target latent, ensuring high consistency; (2) <b>Multi-Reference Disentanglement Loss</b> (green region) maximizes the divergence between different references’ attention distributions, pushing each subject into orthogonal representational subspaces.

## 🚀 Updates
To support research and the open-source community, we will release the entire project—including datasets, inference pipelines, and model weights. Thank you for your patience and continued support! 🌟
- ✅ Release arXiv paper
- ✅ Release codes
- ✅ Release model checkpoints (512x512).
- ⬜ Release model checkpoints (1024x1024).
- ⬜ Release the SemAlign-MS dataset 

<!-- ✅ -->