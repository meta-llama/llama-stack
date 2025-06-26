import React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface LoginButtonProps {
  provider: {
    id: string;
    name: string;
    icon: LucideIcon;
    loginPath: string;
    buttonColor?: string;
  };
  className?: string;
}

export function LoginButton({ provider, className }: LoginButtonProps) {
  const handleLogin = async () => {
    // Add redirect_url parameter to tell backend where to redirect after OAuth
    const redirectUrl = `${window.location.origin}/auth/github/callback`;
    const loginUrl = `${provider.loginPath}?redirect_url=${encodeURIComponent(redirectUrl)}`;
    window.location.href = loginUrl;
  };

  const Icon = provider.icon;

  return (
    <Button
      onClick={handleLogin}
      className={cn(
        "w-full flex items-center justify-center gap-3",
        provider.buttonColor || "bg-gray-900 hover:bg-gray-800",
        className,
      )}
    >
      <Icon className="h-5 w-5" />
      <span>Continue with {provider.name}</span>
    </Button>
  );
}
