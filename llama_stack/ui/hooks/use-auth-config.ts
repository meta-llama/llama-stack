import { useState, useEffect } from "react";

export function useAuthConfig() {
  const [isAuthConfigured, setIsAuthConfigured] = useState(true);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkAuthConfig = async () => {
      try {
        const response = await fetch("/api/auth/github/login", {
          method: "HEAD",
          redirect: "manual",
        });

        setIsAuthConfigured(response.status !== 404);
      } catch (error) {
        console.error("Auth config check error:", error);
        setIsAuthConfigured(true);
      } finally {
        setIsChecking(false);
      }
    };

    checkAuthConfig();
  }, []);

  return { isAuthConfigured, isChecking };
}
