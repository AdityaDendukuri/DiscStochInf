using DiscStochInf
using Documenter

DocMeta.setdocmeta!(DiscStochInf, :DocTestSetup, :(using DiscStochInf); recursive=true)

makedocs(;
    modules=[DiscStochInf],
    authors="Aditya Dendukuri (UCSB)",
    sitename="DiscStochInf.jl",
    format=Documenter.HTML(;
        canonical="https://AdityaDendukuri.github.io/DiscStochInf.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AdityaDendukuri/DiscStochInf.jl",
    devbranch="main",
)
